#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-03-31 13:42:31 (UTC+0200)


import numpy
try:
    import MDAnalysis
    mdanalysis = True
except ImportError:
    mdanalysis = False
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy


def array_to_dcd(array, outfile='traj.dcd'):
    """
    Convert the numpy array to a dcd file trajectory
    """
    if not mdanalysis:
        print "MDAnalysis is not installed, cannot create dcd file"
        return None
    else:
        n_frames = array.shape[0]
        n_atoms = array.shape[1] / 3
        array = array.reshape(n_frames, n_atoms, 3)
        ts =  MDAnalysis.coordinates.base.Timestep(n_atoms)
        W = MDAnalysis.Writer(outfile, n_atoms)
        for f in array:
            ts.positions = f
            W.write_next_timestep(ts)
        return None

def array_to_pdb(array, struct, outfile='traj.pdb', b_factor=None):
    """
    Convert the numpy array to a pdb multimodel file
    If b_factor is not None (typically a n_frames × n_atoms array), the values
    are added to the b-factor column of the output pdb file.
    """
    if not mdanalysis:
        print "MDAnalysis is not installed, cannot create dcd file"
        return None
    else:
        n_frames = array.shape[0]
        n_atoms = array.shape[1] / 3
        array = array.reshape(n_frames, n_atoms, 3)
        u = MDAnalysis.Universe(struct)
        W = MDAnalysis.Writer(outfile, multiframe=True)
        b_factor = numpy.nan_to_num(b_factor)
        for i, f in enumerate(array):
            u.atoms.positions = f
            if b_factor is not None:
                u.atoms.set_bfactors(b_factor[i])
            W.write(u.atoms)
        return u

def get_attribute_assignment_files(array, attribute_name="radius", outfilename="atomic_attributes.txt"):
    """
    Generates a per atom attribute assignment files for chimera as defined in
    https://goo.gl/9OrcCa
    The outputed files can be used to define atomic attributes in chimera with
    the command below:
    defattr atomic_attributes.txt

    For a trajectory, one file per frame can be generated. A Per-Frame command
    could be written to define the attribute for each frame:

    defattr attribute_files/<FRAME>.txt

    for files stored in attribute_files directory and named after the frame id
    starting from 1.

    If the attribute_name is radius the default, the vdw radii will be set on
    the fly when the frame is changing.

    """
    header = "attribute: %s\n"%attribute_name
    header += "match mode: 1-to-1\n"
    header += "recipient: atoms\n"
    data = "\n"
    for i, value in enumerate(array):
        data += "\t@/serialNumber=%d\t%.4g\n"%(i+1, value)
    outfile = open(outfilename, "w")
    outfile.write(header)
    outfile.write(data)
    outfile.close()
    return None

def plot_arrays(gtm, array2=None, scatter=None, scatter_attribute="r.",
                scaling_dim = 3, markersize=4., fig = None,
                ax = None, title=None, xlabel="PC1", ylabel="PC2", levels=100,
                cmap=matplotlib.cm.jet, vmin = None, vmax = None,
                scatter_mode='mode'):
    """
    plot -gtm.log_density as contour and array2 if not None as contourf

    • If scatter is not None (an N*2 matrix): plot the N point as a scatter plot
    • levels: number of contour levels for contourf
    • cmap: color map for the contourf
    • vmin, vmax: set the min and max values for the contourf plot
    • if scatter_mode is 'mode' (default), plot the posterior mode else if
      scatter_mode is 'mean', plot the posterior mean instead
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    n_atoms = gtm.T.shape[1] / scaling_dim
    x1 = 2*numpy.sqrt(gtm.eival[0]*gtm.max_norm**2 / n_atoms)
    x2 = 2*numpy.sqrt(gtm.eival[1]*gtm.max_norm**2 / n_atoms)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    R = numpy.exp(gtm.logR)
    mask = (R.sum(axis=1).reshape(gtm.nx, gtm.ny) ==0)
    if gtm.log_density is None:
        gtm.get_log_density()
    array1 = -gtm.log_density
    array1[mask] = numpy.nan
    extent=(0,x1,0,x2)
    c = ax.contour(array1.T[::-1,:], 10, extent=extent, cmap = matplotlib.cm.gray)
    if scatter_mode == "mode":
        if gtm.posterior_mode is None:
            posterior_m = copy.deepcopy(gtm.get_posterior_mode())
        else:
            posterior_m = copy.deepcopy(gtm.posterior_mode)
    elif scatter_mode == "mean":
        if gtm.posterior_mean is None:
            posterior_m = copy.deepcopy(gtm.get_posterior_mean())
        else:
            posterior_m = copy.deepcopy(gtm.posterior_mean)
        xy = (numpy.sqrt(gtm.eival[0]), numpy.sqrt(gtm.eival[1]))
        posterior_m = (.5*( gtm.posterior_mean + xy ) / xy ) * (gtm.nx, gtm.ny)
    new_basis = lambda x: [-1,1]*([0,x2] - x*numpy.asarray([x1, x2])/numpy.float_(gtm.log_density.shape))
    posterior_m = new_basis(posterior_m)
    ax.plot(posterior_m[:,0], posterior_m[:,1],
            'w.', alpha=.5, markersize=markersize)
    if scatter is not None:
        scatter = new_basis(scatter)
        ax.plot(scatter[:,0], scatter[:,1],
                scatter_attribute, markersize=8*markersize)
    if array2 is None:
        array2 = array1
    c = ax.contourf(array2.T[::-1,:], levels, extent=extent, cmap=cmap,
                    vmin=vmin, vmax=vmax)
    cb = plt.colorbar(c, cax=cax)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax, extent
