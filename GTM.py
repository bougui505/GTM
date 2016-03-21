#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-03-14 15:14:18 (UTC+0100)

import numpy
import scipy.spatial.distance
import scipy.misc
import progress_reporting as Progress

class GTM:
    def __init__(self, inputmat, (nx, ny), n_center = 10, sigma=2, alpha = 0., beta_factor = 2):
        """

        • inputmat: input data size: n×d, with n the number of data and d the 
        dimension of the data -> self.T

        • nx, ny: dimension of the latent space (number of cells)

        • n_center: number of center for the basis functions

        • sigma: radius for the radial basis factors

        • alpha: 1/(variance) of the weight (W)

        • beta_factor: factor to scale the beta initialization (beta_factor =
        (sigma_mapping/sigma_data)**2)

        """
        self.T = inputmat - inputmat.mean(axis=0)
        self.n, self.d = self.T.shape
        # Set automatic size of the array according to the PCA of the input data
        self.nx, self.ny, self.eival, self.eivec = self.get_dim(nx, ny)
        self.T = numpy.dot(self.T, self.eivec)
        self.T /= self.T.max()
        # Grid of the latent space
        self.X = self.get_grid(self.nx, self.ny)
        # Define the radial basis function network
        self.k = self.nx * self.ny
        self.Phi, self.centers, self.sigma = self.radial_basis_function_network(n_center=n_center, sigma=float(sigma))
        # Initial weigths:
        self.W = self.init_weights(self.eivec)
        # Projection of the Latent space to the Data space:
        self.y = numpy.dot(self.Phi, self.W)
        # Align the center of the data space to the center of the projected latent space (y)
        self.T += self.y.mean(axis=0)
        # 1/(variance) of the weight (𝛼)
        self.alpha = alpha
        # Initialize beta (inverse of the variance):
        self.beta = self.init_beta(self.W, factor = beta_factor)
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

    def get_dim(self, x_dim, y_dim, max_input=1000):
        """
        Return the dimension of the Map accordingly to the 2 first principal components of the dataset t
        max_input: maximum number of input data points to tke into account for the PCA
        """
        f = int(self.T.shape[0]/max_input)
        M = self.T[::f]
        covarray = numpy.dot(M.T,M)
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
        X = numpy.asarray(numpy.meshgrid(numpy.linspace(0,1,num=x_dim), numpy.linspace(0,1,num=y_dim))).T
        return X

    def radial_basis_function_network(self, n_center, sigma):
        """
        Return the radial basis function network for a given radius, and the corresponding network X.
        The radius is given in spacing unit: radius×s where s is the spacing of the network
        The number of centers is n_center
        """
        factor = int(numpy.sqrt(self.nx*self.ny/n_center))
        num_x = self.nx/factor
        num_y = self.ny/factor
        centers = numpy.asarray(numpy.meshgrid(numpy.linspace(0,self.nx-1,num_x),
                  numpy.linspace(0,self.ny-1,num_y))).T.reshape(num_x*num_y,2)
        radius = sigma
        m = len(centers)
        Phi = numpy.empty((self.k,m))
        mu_list = []
        for i, center in enumerate(centers):
            center = self.X[tuple(center)]
            mu_list.append(center)
            phi = self.radial_basis_function(center, radius)
            Phi[:,i] = phi(self.X).flatten()
        Phi = numpy.c_[Phi, self.X.reshape(self.k,2), numpy.ones(self.k)]
        return numpy.asarray(Phi), numpy.asarray(mu_list), radius

    def init_weights(self, eivec):
        W = numpy.zeros((self.Phi.shape[1],self.T.shape[1]))
        W[-3:-1,:] = eivec[:,:2].T
        y = numpy.dot(self.Phi,W)
        y_var = y.var(axis=0)
        data_var = self.T.var(axis=0)
        W = W*numpy.sqrt(data_var)/numpy.sqrt(y_var)
        return W

    def init_beta(self, W, factor):
        """
        factor is a scalar to scale the beta value.
        factor = (sigma_mapping/sigma_data)**2
        """
        y = numpy.dot(self.Phi,W)
        sqcdist = scipy.spatial.distance.cdist(y,self.T, metric='sqeuclidean')
        beta = 1/ ( factor*sqcdist.sum()/(numpy.prod(self.T.shape)*self.k) )
        return beta

    def get_likelihood(t_n, x_index, W, beta):
        """
        t_n: on vector of the input space
        x_index: flatten index of the latent space
        W: Weights of the latent space
        beta: variance of the latent space
        """
        D = W.shape[1]
        y = numpy.dot(Phi[x_index], W)
        sqdist = ((y-t_n)**2).sum()
        L = (beta/(2*numpy.pi))**(D/2.)*numpy.exp(-(beta/2)*sqdist)
        return L

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

    def get_evidence(t_n, W, beta):
        D = W.shape[1]
        sqdists = ((numpy.dot(Phi, W) - t_n[:,None])**2).sum(axis=0)
        L = (beta/(2*numpy.pi))**(D/2.)*numpy.exp(-(beta/2)*sqdists)
        E = L.sum()
        return E

    def get_evidence_array(t, W, beta):
        """
        Return the array of evidence: 
        Size: N, with N the number of input data points
        """
        D = W.shape[1]
        sqcdist = scipy.spatial.distance.cdist(numpy.dot(Phi, W), t, 'sqeuclidean')
        L = (beta/(2*numpy.pi))**(D/2.)*numpy.exp(-(beta/2)*sqcdist)
        E = L.sum(axis=0)
        return E

    def get_posterior(t_n, x_index, W, beta):
        """
        t_n: on vector of the input space
        x_index: flatten index of the latent space
        W: Weights of the latent space
        beta: variance of the latent space
        """
        L = get_likelihood(t_n, x_index, W, beta)
        E = get_evidence(t_n, W, beta)
        P = L/E
        return P

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
        return logR, sqcdist, ll

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

    def learn(self, n_iterations):
        log_likelihood = []
        progress = Progress.Progress(n_iterations, delta=10)
        logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
        R = numpy.exp(logR)
        print "Starting Log-likelihood: %.4g"%ll
        for i in range(n_iterations):
            if numpy.isnan(R).any():
                print "There is %d/%d NaN elements in 𝐑. Try to increase beta_factor (noise)..."%(numpy.isnan(R_old).sum(), R_old.size)
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
            log_likelihood.append(ll)
            sigma_mapping = numpy.sqrt(self.d/self.beta)
            sigma_mapping_normalized = sigma_mapping / self.sigma_data
            progress.count(report="𝓵 = %.4g | 𝛽 = %.4g | 𝜎_mapping = %.4g | 𝜎_mapping/𝜎_data = %.4g"%(ll, self.beta, sigma_mapping, sigma_mapping_normalized))
        return self.W, self.beta, log_likelihood

    def posterior_mode(self):
        """
        Return the posterior mode projection:
        x_n = argmax_{x_k}(p(x_k|t_n))
        """
        logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
        R = numpy.exp(logR)
        posterior_mode = numpy.asarray([numpy.unravel_index(e, (self.nx, self.ny)) for e in R.argmax(axis=0)])
        return posterior_mode

    def posterior_mean(self):
        """
        Return the posterior mean projection:
        x_n = sum_k(x_k.p(x_k|t_n))
        """
        logR, sqcdist, ll = self.get_posterior_array(self.T, self.W, self.beta)
        R = numpy.exp(logR)
        posterior_mean = numpy.dot(R.T, self.X.reshape(self.nx * self.ny, 2))
        return posterior_mean
