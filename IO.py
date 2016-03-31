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
