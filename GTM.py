#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-03-14 15:14:18 (UTC+0100)

import numpy
import scipy.spatial.distance
import scipy.misc
import progress_reporting as Progress
import pickle
try:
    import MDAnalysis
    mdanalysis = True
except ImportError:
    mdanalysis = False

class GTM:
    def __init__(self, inputmat, (nx, ny), n_center = 10, sigma=None, alpha = 0.):
        """

        • inputmat: input data size: n×d, with n the number of data and d the 
        dimension of the data -> self.T

        • nx, ny: dimension of the latent space (number of cells)

        • n_center: number of center for the basis functions

        • sigma: radius for the radial basis factors

        • alpha: 1/(variance) of the weight (W)

        """
        self.input_mean = inputmat.mean(axis=0)
        self.T = inputmat - self.input_mean
        self.max_norm = numpy.linalg.norm(self.T, axis=1).max()
        self.T /= self.max_norm
        self.n, self.d = self.T.shape
        # Set automatic size of the array according to the PCA of the input data
        self.nx, self.ny, self.eival, self.eivec = self.get_dim(nx, ny)
        self.T = numpy.dot(self.T, self.eivec)
        # Grid of the latent space
        self.X = self.get_grid(self.nx, self.ny)
        # Define the radial basis function network
        self.k = self.nx * self.ny
        self.Phi, self.centers, self.sigma = self.radial_basis_function_network(n_center=n_center, sigma=sigma)
        # Initial weigths:
        self.W = self.init_weights(self.eivec)
        # Projection of the Latent space to the Data space:
        self.y = numpy.dot(self.Phi, self.W)
        # 1/(variance) of the weight (𝛼)
        self.alpha = alpha
        # Initialize beta (inverse of the variance):
        self.beta = self.init_beta()
        # Give informations about the initial likelihood:
        logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
        print "𝙏: %s"%str(self.T.shape)
        print "𝑿: %s"%str(self.X.shape)
        print "𝜱: %s"%str(self.Phi.shape)
        print "𝞵: %s"%str(self.centers.shape)
        print "𝜎 = %.4g"%self.sigma
        print "𝑾: %s"%str(self.W.shape)
        print "𝛼 = %.4g"%self.alpha
        sigma_mapping = numpy.sqrt(self.d/self.beta)
        self.sigma_data = numpy.linalg.norm(self.T.std(axis=0))
        sigma_mapping_normalized = sigma_mapping / self.sigma_data
        print "𝛽 = %.4g | 𝜎_mapping = %.4g | 𝜎_mapping/𝜎_data = %.4g"%(self.beta, sigma_mapping, sigma_mapping_normalized)
        print "𝓵 = %.4g"%ll
        self.log_density = None
        self.log_likelihood = []
        self.posterior_mode = None
        self.posterior_mean = None

    def get_dim(self, x_dim, y_dim):
        """
        Return the dimension of the Map accordingly to the 2 first principal components of the dataset t
        max_input: maximum number of input data points to tke into account for the PCA
        """
        M = self.T
        covarray = numpy.cov(M.T)
        eival, eivec = numpy.linalg.eigh(covarray)
        args = eival.argsort()[::-1]
        eival = eival[args]
        eivec = eivec[:, args]
        sqev = numpy.sqrt(eival[:2])
        x_dim, y_dim = map(lambda x: int(round(x)), sqev / (
                           (numpy.prod(sqev) / (x_dim * y_dim)) ** (1. / 2)))
        return x_dim, y_dim, eival, eivec

    def radial_basis_function(self, center, radius):
        radius=numpy.float(radius)
        beta = (1/radius**2)
        return lambda x: numpy.exp(-(beta/2) * ((x - center)**2).sum(axis=-1))

    def get_grid(self, x_dim, y_dim):
        x_norm, y_norm = numpy.sqrt(self.eival[0]), numpy.sqrt(self.eival[1])
        X = numpy.asarray(numpy.meshgrid(numpy.linspace(-x_norm, x_norm, num=x_dim), numpy.linspace(-y_norm, y_norm, num=y_dim))).T
        return X

    def radial_basis_function_network(self, n_center, sigma):
        """
        Return the radial basis function network for a given radius, and the corresponding network X.
        The radius is given in spacing unit: radius×s where s is the spacing of the network
        The number of centers is n_center
        """
        h, w, d =  self.X.shape
        n = n_center
        n_x = round(numpy.sqrt(w*n/h + (w-h)**2/(4*h**2)) - (w-h)/(2*h))
        n_y = round(n/n_x)
        delta_x = h/n_y
        delta_y = w/n_x
        centers = []
        for x in numpy.arange(delta_x/2,h+delta_x/2,delta_x):
            for y in numpy.arange(delta_y/2,w+delta_y/2,delta_y):
                centers.append((int(round(x)-1), int(round(y)-1)))
        if sigma is not None:
            radius = sigma
        else:
            radius = 2*numpy.sqrt(self.eival[0])/n_y
        m = len(centers)
        Phi = numpy.empty((self.k,m))
        mu_list = []
        for i, center in enumerate(centers):
            center = self.X[center]
            mu_list.append(center)
            phi = self.radial_basis_function(center, radius)
            Phi[:,i] = phi(self.X).flatten()
        Phi = numpy.c_[Phi, self.X.reshape(self.k,2), numpy.ones(self.k)]
        return numpy.asarray(Phi), numpy.asarray(mu_list), radius

    def init_weights(self, eivec):
        W = numpy.zeros((self.Phi.shape[1],self.T.shape[1]))
        W[-3:-1,:2] = numpy.identity(2)
        return W

    def init_beta(self):
        """
        """
        if len(self.eival) > 2:
            beta = 1/max((numpy.linalg.norm(self.y[1] - self.y[0])/2)**2, self.eival[2])
        else:
            beta = 1/(numpy.linalg.norm(self.y[1] - self.y[0])/2)**2
        return beta

    def get_likelihood_array(self, t, W, beta):
        """
        the likelihood matrix (L) is an array of shape K×N, 
        with K the number of neurons and N the number of input data point.
        """
        D = W.shape[1]
        y = numpy.dot(self.Phi, W)
        sqcdist = scipy.spatial.distance.cdist(numpy.dot(self.Phi, W), t, 'sqeuclidean')
        L = ((beta/(2*numpy.pi))**(D/2.))*numpy.exp(-(beta/2)*sqcdist)
        return L

    def get_log_likelihood(self, L):
        K = L.shape[0] # Number of cells (neurons)
        ll = numpy.log(L.sum(axis=0)/K).sum() # Log likelihood
        return ll

    def get_posterior_array(self, t, W, beta):
        """
        the posterior matrix (R) is an array of shape K×N, 
        with K the number of neurons and N the number of input data point.
        """
        D = W.shape[1]
        y = numpy.dot(self.Phi, W)
        sqcdist = scipy.spatial.distance.cdist(y, t, 'sqeuclidean')
        logL = -(beta/2)*sqcdist
        logE = scipy.misc.logsumexp(-beta/2 * sqcdist, axis=0)
        logR = logL - logE
        # Real value of log-likelihood
        #ll = self.n*numpy.log( (1./self.k) * (beta/(2*numpy.pi))**(self.d/2.)) + logE.sum()
        # Tracking value of log-likelihood
        ll = (self.n*self.d/2)*numpy.log(beta/(2*numpy.pi)) + logE.sum()
        self.logR, self.sqcdist, self.ll = logR, sqcdist, ll
        return logR, sqcdist, ll

    def get_log_density(self, t, W, beta):
        """
        Density estimation of the data in the latent space
        """
        logR, sqcdist, ll = self.logR, self.sqcdist, self.ll
        logE = (self.d/2)*numpy.log(beta/(2*numpy.pi)) + \
                scipy.misc.logsumexp((-beta/2) * sqcdist, axis=0) - \
                 numpy.log(self.k) # evidence (p(t))
        log_density = scipy.misc.logsumexp(logR+logE, axis=1)
        log_density -= scipy.misc.logsumexp(log_density) # normalization factor such as density.sum() == 1
        self.log_density = log_density.reshape((self.nx, self.ny))
        return self.log_density

    def project_data(self, data):
        """
        Project the data onto the map
        """
        logR, sqcdist, ll = self.logR, self.sqcdist, self.ll
        R = numpy.exp(logR)
        data_map = (R*data).sum(axis=1)/ R.sum(axis=1)
        return data_map.reshape((self.nx, self.ny))

    def get_G_array(self, logR):
        """
        Diagonal matrix of size K×K
        input:
        - R: matrix returned by get_posterior_array
        """
        K = logR.shape[0]
        G = numpy.zeros((K,K))
        numpy.fill_diagonal(G, numpy.exp(scipy.misc.logsumexp(logR, axis=1)))
        return G

    def learn(self, n_iterations, report_interval = 10, report_rmsd = False):
        """
        if report_rmsd: report the RMSD in Angstrom for protein mapping only
        """
        progress = Progress.Progress(n_iterations, delta = report_interval)
        logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
        R = numpy.exp(logR)
        print "Starting Log-likelihood: %.4g"%ll
        for i in range(n_iterations):
            if numpy.isnan(R).any():
                print "There is %d/%d NaN elements in 𝐑."%(numpy.isnan(R_old).sum(), R_old.size)
                break
            logbeta = numpy.log(self.n*self.d) - scipy.misc.logsumexp(logR+numpy.log(sqcdist))
            beta = numpy.exp(logbeta)
            G = self.get_G_array(logR)
            lambda_factor = self.alpha / self.beta
            Phi_G_Phi = numpy.dot(self.Phi.T, numpy.dot(G,self.Phi)) - lambda_factor * numpy.identity(self.Phi.shape[1])
            prod_1 = numpy.dot(numpy.linalg.inv(Phi_G_Phi), self.Phi.T)
            prod_2 = numpy.dot(prod_1, R)
            W = numpy.dot(prod_2, self.T) # W new
            self.W = W
            self.beta = beta
            logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
            R = numpy.exp(logR)
            self.log_likelihood.append(ll)
            sigma_mapping = numpy.sqrt(self.d/self.beta)
            sigma_mapping_normalized = sigma_mapping / self.sigma_data
            sigma_w = numpy.linalg.norm(self.W, axis=1).var()
            report="𝓵 = %.4g | 𝛽 = %.4g | 𝜎_mapping = %.4g | 𝜎_mapping/𝜎_data = %.4g | 𝜎_𝑾 = %.4g"%(ll, self.beta, sigma_mapping, sigma_mapping_normalized, sigma_w)
            if report_rmsd:
                rmsd = numpy.sqrt(3/self.beta) * self.max_norm
                report += " | RMSD = %.4g Å"%rmsd
            progress.count(report=report)
        return self.W, self.beta, self.log_likelihood

    def save_data(self, outfile='gtm.dat'):
        data = self.__dict__
        f = open(outfile,'wb')
        pickle.dump(data, f, 2)
        f.close()

    def load_data(self, infile='gtm.dat'):
        data_dict = numpy.load(infile)
        #keys: ['sqcdist', 'Phi', 'logR', 'd', 'sigma_data', 'll', 'eival', 'centers', 'nx', 'ny', 'beta', 'eivec', 'T', 'W', 'X', 'y', 'alpha', 'n', 'sigma', 'k']:
        self.sqcdist = data_dict['sqcdist']
        self.Phi = data_dict['Phi']
        self.logR = data_dict['logR']
        self.d = data_dict['d']
        self.sigma_data = data_dict['sigma_data']
        self.ll = data_dict['ll']
        self.eival = data_dict['eival']
        self.centers = data_dict['centers']
        self.nx = data_dict['nx']
        self.ny = data_dict['ny']
        self.beta = data_dict['beta']
        self.eivec = data_dict['eivec']
        self.T = data_dict['T']
        self.W = data_dict['W']
        self.X = data_dict['X']
        self.y = data_dict['y']
        self.alpha = data_dict['alpha']
        self.n = data_dict['n']
        self.sigma = data_dict['sigma']
        self.k = data_dict['k']
        self.max_norm = data_dict['max_norm']
        self.input_mean = data_dict['input_mean']
        self.log_likelihood = data_dict['log_likelihood']
        self.posterior_mode = data_dict['posterior_mode']
        self.posterior_mean = data_dict['posterior_mean']
        self.log_density = data_dict['log_density']

    def get_posterior_mode(self):
        """
        Return the posterior mode projection:
        x_n = argmax_{x_k}(p(x_k|t_n))
        """
        logR, sqcdist, ll = self.logR, self.sqcdist, self.ll
        R = numpy.exp(logR)
        self.posterior_mode = numpy.asarray([numpy.unravel_index(e, (self.nx, self.ny)) for e in R.argmax(axis=0)])
        return self.posterior_mode

    def get_posterior_mean(self):
        """
        Return the posterior mean projection:
        x_n = sum_k(x_k.p(x_k|t_n))
        """
        logR, sqcdist, ll = self.logR, self.sqcdist, self.ll
        R = numpy.exp(logR)
        self.posterior_mean = numpy.dot(R.T, self.X.reshape(self.nx * self.ny, 2))
        return self.posterior_mean

    def local_std(self):
        """
        get the local standard deviation values for the full map in the data unit
        You hav to divide by sqrt(n_atoms) to obtain an RMSD in angstrom
        """
        return numpy.sqrt((self.project_data(self.sqcdist)) * self.max_norm**2)

    def map_to_data(self):
        """
        Project the latent space to the data space
        """
        self.eivec_inv = numpy.linalg.inv(self.eivec)
        data_proj = numpy.dot(self.Phi, self.W).dot(self.eivec_inv)
        data_proj = data_proj * self.max_norm + self.input_mean
        return data_proj

    def get_atomic_fluctuation(self):
        """

        Return the fluctuation per atom for each structure of the latent space.
        Return an array of size (self.k, self.d/3) where self.d/3 is the number
        of atom.

        """
        y = self.map_to_data()
        n_atoms = self.T.shape[1]/3
        y = y.reshape(self.k, n_atoms,3)
        t = self.T.dot(self.eivec_inv) * self.max_norm + self.input_mean
        t = t.reshape(self.n, n_atoms, 3)
        atomic_fluctuations = []
        for atom_id in range(n_atoms):
            fluctuation = self.project_data(scipy.spatial.distance.cdist(y[:,atom_id,:], t[:,atom_id,:], metric='sqeuclidean'))
            fluctuation = numpy.sqrt(fluctuation)
            atomic_fluctuations.append(fluctuation.flatten())
        atomic_fluctuations = numpy.asarray(atomic_fluctuations).T
        return atomic_fluctuations

    def get_transition_matrix(self):
        """
        return the transition matrix. Relevant only for the clustering of a MD
        simulation...
        Can be used to find kinetic communities using the Graph module:
            graph = Graph.Graph(adjacency_matrix=-numpy.log(P))
            graph.best_partition()
        """
        R = numpy.exp(self.logR)
        P = R[:,:self.n-1].dot(R[:,1:].T) / R.sum(axis=1) # Transition matrix
        P = numpy.nan_to_num(P)
        return P
