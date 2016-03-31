#!/usr/bin/env python
# -*- coding: UTF8 -*-
"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 2015 06 25
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""

import sys;

sys.path.append('.')
from plotdialog import PlotDialog
from plotdialog import RMSD
from plotdialog import SelectClusterMode
from plotdialog import Density
from plotdialog import Data_driven_clustering
from plotdialog import Projection
from plotdialog import Add_experimental_data
import numpy
import Combine
from chimera import update
from chimera import openModels
from chimera import numpyArrayFromAtoms
from chimera.match import matchPositions
from Movie.analysis import analysisAtoms, AnalysisError
from collections import OrderedDict
import Midas
import matplotlib
import pickle
import chimera

from chimera import runCommand



class UmatPlot(PlotDialog):
    def __init__(self, movie):
        self.movie = movie
        self.data = numpy.load('gtm.dat')
        self.k = self.data['k'] # number of cells of the map
        self.nx = self.data['nx'] # Number of cells of the map for dimension x
        self.ny = self.data['ny'] # Number of cells of the map for dimension y
        self.logR = self.data['logR']
        R = numpy.exp(self.logR)
        mask = (R.sum(axis=1).reshape(self.nx, self.ny) ==0)
        self.matrix = -self.data['log_density']
        self.matrix[mask] = numpy.nan
        self.min_value = numpy.nanmin(self.matrix)
        PlotDialog.__init__(self, self.min_value, numpy.nanmax(self.matrix))
        self.master = self._master
        self.displayed_matrix = self.matrix
        self.selection_mode = 'Cell'
        self.bmus = numpy.asarray([numpy.unravel_index(e, (self.nx,self.ny))
                                    for e in range(self.k)])
        self.selected_neurons = OrderedDict([])
        self.colors = []  # colors of the dot in the map
        self.subplot = self.add_subplot(1, 1, 1)
        self.colorbar = None
        self.cluster_map = None
        self.highlighted_cluster = None
        self._displayData()
        movie.triggers.addHandler(self.movie.NEW_FRAME_NUMBER, self.update_bmu, None)
        self.registerPickHandler(self.onPick)
        self.figureCanvas.mpl_connect("key_press_event", self.onKey)
        self.figureCanvas.mpl_connect("key_release_event", self.offKey)
        self.keep_selection = False
        self.init_models = set(openModels.list())
        self.movie_id = movie.model.id
        self.i, self.j = None, None  # current neuron
        self.rmsd_list = None
        self.rep_rmsd = None
        self.experimental_data_filename = None
        self.data_driven_clusters = None
        self.ctrl_pressed = False
        self.motion_notify_event = None
        self.slice_id = 0 # slice of the matrix to display for high dimensional data
        self.plot1D = None # 1D plot for multidimensional features
        self.clustermode = (1,'Frames') # to display either Density map or ensemble of frames
        self.rep_map = numpy.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                self.rep_map[i,j] = numpy.ravel_multi_index((i,j), (self.nx, self.ny))

    def switch_matrix(self, value):
        if self.display_option.getvalue() == "Density" or self.display_option.getvalue() is None:
            self.displayed_matrix = self.matrix
        self._displayData()

    def update_selection_mode(self, value):
        """

        Change the mouse behaviour for selection

        """
        self.selection_mode = self.selection_mode_menu.getvalue() # 'Cell' or 'Cluster'
        if self.selection_mode == 'Cell':
            self.figureCanvas.mpl_disconnect(self.motion_notify_event)
            self.highlighted_cluster = None
        elif self.selection_mode == 'Cluster':
            #dialog box to select cluster mode
            dlg = SelectClusterMode(self.movie)  
            params=dlg.run(self.master)
            if params is not None:
                self.clustermode = params
            self.motion_notify_event = self.figureCanvas.mpl_connect("motion_notify_event", self.highlight_cluster)

    def get_clusters(self, value):
        """
        Define clusters with the threshold given by the slider dialog (Cluster())
        """
        threshold = self.slider.get()
        self.cluster_map = self.matrix <= threshold
        self._displayData()

    def update_bmu(self, event_name, empty, frame_id):
        bmu = self.bmus[frame_id - 1]
        y, x = bmu
        y += .5
        x += .5
        if not self.keep_selection:
            ax = self.subplot
            ax.clear()
            ax.scatter(x, y, c='r', edgecolors='white')
            nx, ny = self.matrix.shape
            if len(self.displayed_matrix.shape) == 2: # two dimensional array
                ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
            else: # we must slice the matrix
                ax.imshow(self.displayed_matrix[:,:,self.slice_id], interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
            if self.cluster_map is not None:
                ax.contour(self.cluster_map, 1, colors='white', linewidths=2.5, extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
                ax.contour(self.cluster_map, 1, colors='red', extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
            self.figure.canvas.draw()

    def _displayData(self):
        ax = self.subplot
        ax.clear()
        for i, neuron in enumerate(self.selected_neurons):
            y, x = neuron
            y += .5
            x += .5
            if self.keep_selection:
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
            elif not self.keep_selection:
                ax.scatter(x, y, c=self.colors[i], edgecolors='white')
        nx, ny = self.matrix.shape
        heatmap = ax.imshow(self.displayed_matrix, interpolation='nearest', extent=(0, ny, nx, 0), picker=True)
        if self.cluster_map is not None:
            ax.contour(self.cluster_map, 1, colors='white', linewidths=2.5, extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
            ax.contour(self.cluster_map, 1, colors='red', extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
        if self.highlighted_cluster is not None:
            ax.contour(self.highlighted_cluster, 1, colors='white', linewidths=2.5,
                        extent=(0, ny, 0, nx), origin='lower') # display the contours for cluster
            ax.contour(self.highlighted_cluster, 1, colors='green', extent=(0, ny, 0, nx),
                        origin='lower') # display the contours for cluster

        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(heatmap)
        else:
            self.colorbar.update_bruteforce(heatmap)
        self.figure.canvas.draw()

    def close_current_models(self):
        self.selected_neurons = OrderedDict([])
        self.colors = []
        current_models = set(openModels.list())
        models_to_close = current_models - self.init_models
        openModels.close(models_to_close)
        update.checkForChanges()  # to avoid memory leaks (see: http://www.cgl.ucsf.edu/chimera/docs/ProgrammersGuide/faq.html)

    def display_frame(self, frame_id):
        self.movie.currentFrame.set(frame_id)
        self.movie.LoadFrame()

    def add_model(self, name):
        mol = self.movie.model.Molecule()
        Combine.cmdCombine([mol], name=name)

    def update_model_color(self):
        model_id = openModels.listIds()[-1][0]
        if not self.keep_selection:
            Midas.color('orange red', '#%d' % self.movie_id)
        else:
            Midas.color('forest green', '#%d' % model_id)
            Midas.color('byhet', '#%d' % model_id)
            Midas.color('forest green', '#%d' % self.movie_id)
        Midas.color('byhet', '#%d' % self.movie_id)

    def onPick(self, event):
        if self.selection_mode == 'Cell': # Select unique cells
            if event.mouseevent.button == 3 or self.ctrl_pressed:
                self.keep_selection = True
            else:
                self.keep_selection = False

            if not self.keep_selection:
                self.close_current_models()
            else:
                if len(self.selected_neurons) == 1:
                    if self.i is not None and self.j is not None:
                        self.add_model(name='%d,%d' % (self.i, self.j))
            if event.mouseevent.button == 1 or event.mouseevent.button == 3:
                x, y = event.mouseevent.xdata, event.mouseevent.ydata
                self.j, self.i = int(x), int(y)
                if (self.i, self.j) not in self.selected_neurons.keys():
                    frame_id = self.rep_map[self.i, self.j] + 1
                    if not numpy.isnan(frame_id):
                        frame_id = int(frame_id)
                        if self.keep_selection:
                            self.colors.append('g')
                        else:
                            self.colors.append('r')
                        self.display_frame(frame_id)
                        if self.keep_selection:
                            self.add_model(name='%d,%d' % (self.i, self.j))
                        self.selected_neurons[(self.i, self.j)] = openModels.list()[-1]
                        self.update_model_color()
                else:
                    model_to_del = self.selected_neurons[(self.i, self.j)]
                    if model_to_del not in self.init_models:
                        openModels.close([model_to_del])
                        del self.selected_neurons[(self.i, self.j)]
                self.get_basin(None) # to display the basin around the selected cell
                #self._displayData() # commented as it's already done by self.get_basin(None) above
        elif self.selection_mode == 'Cluster' and event.mouseevent.button == 1:
            self.close_current_models()
            if self.highlighted_cluster is not None:
                self.highlighted_cluster[numpy.isnan(self.highlighted_cluster)] = False
                self.highlighted_cluster = numpy.bool_(self.highlighted_cluster)
                frame_ids = self.rep_map[self.highlighted_cluster] + 1
                frame_ids = frame_ids[~numpy.isnan(frame_ids)]
                n = len(frame_ids)
                if self.clustermode[1] == "Frames":
                    if n > 10:
                        frame_ids = frame_ids[::n/10] # take only ten representatives
                    for frame_id in frame_ids:
                        frame_id = int(frame_id)
                        self.display_frame(frame_id)
                        self.add_model(name='cluster')
                elif self.clustermode[1] == 'Density':
                    if n > 100:
                        frame_ids = frame_ids[::n/100] # take only 100 representatives
                    trajMol = self.movie.model._mol

                    if chimera.selection.currentEmpty():
                        #something to select all atoms
                        runCommand("select #%d" % self.movie_id)
                    atoms = [a for a in chimera.selection.currentAtoms() if a.molecule == trajMol]    
                    name="ClusterDensityMap"
                    self.computeVolume(atoms, frame_ids=frame_ids,volumeName=name, spacing=self.clustermode[0])
                    model_id=openModels.listIds()[-1][0]
                    #Midas.color('aquamarine,s', '#%d' %model_id)
                    runCommand("volume #%d level 50. color aquamarine style surface" %model_id)



    def highlight_cluster(self, event):
        x, y = event.xdata, event.ydata
        #threshold = self.slider2.get() # threshold for slider 2
        threshold = 0
        if x is not None and y is not None:
            self.j, self.i = int(x), int(y)
            if self.fold.has_key((self.i,self.j)):
                if threshold > 0:
                    self.get_basin(None, display=False)
                    self.highlighted_cluster = self.cluster_map
                else:
                    cell = self.fold[(self.i, self.j)]
                    if self.cluster_map[self.i,self.j]:
                        self.highlighted_cluster = self.pick_up_cluster((self.i,self.j))
                    else:
                        self.highlighted_cluster = None
                self._displayData()

    def neighbor_dim2_toric(self, p, s):
        """Efficient toric neighborhood function for 2D SOM.
        """
        x, y = p
        X, Y = s
        xm = (x - 1) % X
        ym = (y - 1) % Y
        xp = (x + 1) % X
        yp = (y + 1) % Y
        return [(xm, ym), (xm, y), (xm, yp), (x, ym), (x, yp), (xp, ym), (xp, y), (xp, yp)]

    def get_neighbors_of_area(self, cluster_map):
        """
        return the neighboring indices of an area defined by a boolean array
        """
        neighbors = []
        shape = cluster_map.shape
        for cell in numpy.asarray(numpy.where(cluster_map)).T:
            for e in self.neighbor_dim2_toric(cell, shape):
                if not cluster_map[e]:
                    neighbors.append(e)
        return neighbors


    def pick_up_cluster(self, starting_cell):
        """

        • pick up a cluster according to connexity for U-matrix based clustering

        • For data_driven clusters: pick up cluster from chi values of
        self.data_driven_clusters

        """
        data_driven_selection = False
        if self.data_driven_clusters is not None:
            if (self.cluster_map == (self.data_driven_clusters != 0)).all():
                data_driven_selection = True
        if not data_driven_selection:
            cluster_map = self.fold_matrix(self.cluster_map)
            cell = self.fold[starting_cell]
            visit_mask = numpy.zeros(self.init_som_shape, dtype=bool)
            visit_mask[cell] = True
            checkpoint = True
            while checkpoint:
                checkpoint = False
                for e in self.get_neighbors_of_area(visit_mask):
                    if cluster_map[e]:
                        visit_mask[e] = True
                        checkpoint = True
            return self.unfold_matrix(visit_mask)
        else: # Data driven clusters
            return self.data_driven_clusters == self.data_driven_clusters[starting_cell]


    def onKey(self, event):
        if event.key == 'control':
            self.ctrl_pressed = True

    def offKey(self, event):
        if event.key == 'control':
            self.ctrl_pressed = False

    def get_basin(self, value, display=True):
        """
        Define basin with the threshold given by the slider dialog
        """
        #threshold = self.slider2.get()
        threshold = 0
        if self.i is not None and self.j is not None\
            and threshold > 0 and self.fold.has_key((self.i, self.j)):
            cell = self.fold[(self.i, self.j)]
            self.cluster_map = self.unfold_matrix(self.dijkstra(starting_cell=cell, threshold=threshold) != numpy.inf)
        else:
            self.get_clusters(None)
        if display:
            self._displayData()

    def dijkstra(self, starting_cell = None, threshold = numpy.inf):
        """

        Apply dijkstra distance transform to the SOM map.
        threshold: interactive threshold for local clustering

        """
        ms_tree = self.minimum_spanning_tree
        nx, ny = self.init_som_shape
        nx2, ny2 = ms_tree.shape
        visit_mask = numpy.zeros(nx2, dtype=bool)
        m = numpy.ones(nx2) * numpy.inf
        if starting_cell is None:
            cc = numpy.unravel_index(ms_tree.argmin(), (nx2, ny2))[0]  # current cell
        else:
            cc = numpy.ravel_multi_index(starting_cell, (nx, ny))
        m[cc] = 0
        while (~visit_mask).sum() > 0:
            neighbors = [e for e in numpy.where(ms_tree[cc] != numpy.inf)[0] if not visit_mask[e]]
            for e in neighbors:
                d = ms_tree[cc, e] + m[cc]
                if d < m[e]:
                    m[e] = d
            visit_mask[cc] = True
            m_masked = numpy.ma.masked_array(m, visit_mask)
            cc = m_masked.argmin()
            if m[m != numpy.inf].max() > threshold:
                break
        return m.reshape((nx, ny))

    
    def computeVolume(self, atoms, frame_ids, volumeName=None, spacing=0.5, radiiTreatment="ignored"):
        #function taken from Movie/gui.py and tweaked to compute volume based on an array of frame_ids
        from Matrix import xform_matrix
        gridData = {}
        from math import floor
        from numpy import array, float32, concatenate
        from _contour import affine_transform_vertices
        insideDeltas = {}
        include = {}
        sp2 = spacing * spacing
        for fn in frame_ids:
            cs = self.movie.findCoordSet(fn)
            if not cs:
                self.movie.status("Loading frame %d" % fn)
                self.movie._LoadFrame(int(fn), makeCurrent=False)
                cs = self.movie.findCoordSet(fn)

            self.movie.status("Processing frame %d" % fn)
            pts = array([a.coord(cs) for a in atoms], float32)
            if self.movie.holdingSteady:
                if bound is not None:
                    steadyPoints = array([a.coord(cs)
                        for a in steadyAtoms], float32)
                    closeIndices = find_close_points(
                        BOXES_METHOD, steadyPoints,
                        #otherPoints, bound)[1]
                        pts, bound)[1]
                    pts = pts[closeIndices]
                try:
                    xf, inv = self.movie.transforms[fn]
                except KeyError:
                    xf, inv = self.movie.steadyXform(cs=cs)
                    self.movie.transforms[fn] = (xf, inv)
                xf = xform_matrix(xf)
                affine_transform_vertices(pts, xf)
                affine_transform_vertices(pts, inverse)
            if radiiTreatment != "ignored":
                ptArrays = [pts]
                for pt, radius in zip(pts, [a.radius for a in atoms]):
                    if radius not in insideDeltas:
                        mul = 1
                        deltas = []
                        rad2 = radius * radius
                        while mul * spacing <= radius:
                            for dx in range(-mul, mul+1):
                                for dy in range(-mul, mul+1):
                                    for dz in range(-mul, mul+1):
                                        if radiiTreatment == "uniform" \
                                        and min(dx, dy, dz) > -mul and max(dx, dy, dz) < mul:
                                            continue
                                        key = tuple(sorted([abs(dx), abs(dy), abs(dz)]))
                                        if key not in include.setdefault(radius, {}):
                                            include[radius][key] = (dx*dx + dy*dy + dz*dz
                                                    ) * sp2 <= rad2
                                        if include[radius][key]:
                                            deltas.append([d*spacing for d in (dx,dy,dz)])
                            mul += 1
                        insideDeltas[radius] = array(deltas)
                        if len(deltas) < 10:
                            print deltas
                    if insideDeltas[radius].size > 0:
                        ptArrays.append(pt + insideDeltas[radius])
                pts = concatenate(ptArrays)
            # add a half-voxel since volume positions are
            # considered to be at the center of their voxel
            from numpy import floor, zeros
            pts = floor(pts/spacing + 0.5).astype(int)
            for pt in pts:
                center = tuple(pt)
                gridData[center] = gridData.get(center, 0) + 1

        # generate volume
        self.movie.status("Generating volume")
        axisData = zip(*tuple(gridData.keys()))
        minXyz = [min(ad) for ad in axisData]
        maxXyz = [max(ad) for ad in axisData]
        # allow for zero-padding on both ends
        dims = [maxXyz[axis] - minXyz[axis] + 3 for axis in range(3)]
        from numpy import zeros, transpose
        volume = zeros(dims, int)
        for index, val in gridData.items():
            adjIndex = tuple([index[i] - minXyz[i] + 1
                            for i in range(3)])
            volume[adjIndex] = val
        from VolumeData import Array_Grid_Data
        gd = Array_Grid_Data(volume.transpose(),
                    # the "cushion of zeros" means d-1...
                    [(d-1) * spacing for d in minXyz],
                    [spacing] * 3)
        if volumeName is None:
            volumeName = self.movie.ensemble.name
        gd.name = volumeName

        # show volume
        self.movie.status("Showing volume")
        import VolumeViewer
        dataRegion = VolumeViewer.volume_from_grid_data(gd)
        vd = VolumeViewer.volumedialog.volume_dialog(create=True)
        vd.message("Volume can be saved from File menu")
        self.movie.status("Volume shown")

from chimera.extension import manager
from Movie.gui import MovieDialog

movies = [inst for inst in manager.instances if isinstance(inst, MovieDialog)]
if len(movies) != 1:
    raise AssertionError("not exactly one MD Movie")
movie = movies[0]
UmatPlot(movie)
