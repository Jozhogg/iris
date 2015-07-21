# (C) British Crown Copyright 2013 - 2015, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Regridding functions.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range)  # noqa

from collections import namedtuple
import copy
import warnings

import numpy as np
import numpy.ma as ma
from scipy.sparse import csc_matrix
from scipy import spatial

import iris.analysis.cartography
from iris.analysis._interpolation import get_xy_dim_coords, snapshot_grid

import iris.coord_systems
import iris.cube
import iris.unit
import cartopy.crs as ccrs


_Version = namedtuple('Version', ('major', 'minor', 'micro'))
_NP_VERSION = _Version(*(int(val) for val in
                         np.version.version.split('.')))

class BackprojectRegridder(object):
    """
    TODO: Add doc

    """
    def __init__(self, src_grid_cube, target_grid_cube, weights):
        """
        Create a regridder for conversions between the source
        and target grids.

        Args:

        * src_grid_cube:
            The :class:`~iris.cube.Cube` providing the source grid.
        * tgt_grid_cube:
            The :class:`~iris.cube.Cube` providing the target grid.
        * weights:
            A :class:`numpy.ndarray` instance that defines the weights
            for the grid cells of the source grid. Must have the same shape
            as the data of the source grid.

        """
        # Validity checks.
        if not isinstance(src_grid_cube, iris.cube.Cube):
            raise TypeError("'src_grid_cube' must be a Cube")
        if not isinstance(target_grid_cube, iris.cube.Cube):
            raise TypeError("'target_grid_cube' must be a Cube")
        # Snapshot the state of the cubes to ensure that the regridder
        # is impervious to external changes to the original source cubes.
        self._src_cube = src_grid_cube.copy()
        self._tgt_cube = target_grid_cube.copy()
        self.weights = weights
        
        # Instantiate a regridder for the given source and target cubes
        
        # Get the source cube's cell grid
        (self.src_grid, 
        self.x_points, 
        self. y_points) = BackprojectRegridder.cube_grid(self._src_cube)
        
        self.tgt_grid = BackprojectRegridder.cube_grid(self._tgt_cube)[0]

        # calculate x and y midpoints of source cell grid
        cellx = np.zeros((self.src_grid[0].shape[0]-1, 
                                    self.src_grid[0].shape[1]-1))
        celly = np.zeros((self.src_grid[0].shape[0]-1, 
                                    self.src_grid[0].shape[1]-1))

        for i in range(self.src_grid[0].shape[1]-1):
        
            cellx[:, i] = float(self.src_grid[0][0,i] 
                                + self.src_grid[0][0,i+1])/2
                                
        for i in range(self.src_grid[0].shape[0]-1):
        
            celly[i, :] = float(self.src_grid[1][i,0] 
                                + self.src_grid[1][i+1,0])/2
        
        #Create CKDTree of centre points of cells
        tree_grid = zip(cellx.ravel(), celly.ravel())
        self.tree = spatial.cKDTree(tree_grid)
    
    @staticmethod
    def cube_grid(cube):
        """
        Creates the bounding grid for a given cube and returns with the x and
        y coordinates of the points

        Args:

        * cube:
            An instance of :class:`iris.cube.Cube`.

        Returns:
            A tuple of the grid, x coordinate array and y coordinate array

        """
        
        # Guess bounds if necessary
        if cube.coord(axis ='y').bounds == None:
            cube.coord(axis = 'y').guess_bounds()
        if cube.coord(axis = 'x').bounds == None:
            cube.coord(axis = 'x').guess_bounds()
        # Convert long bounds to a 1D array
        long_size = cube.coord(axis = 'x').bounds.shape[0]  
        long_nums = cube.coord(axis = 'x').bounds[:,0].reshape(long_size,1)
        long_end = np.asarray(cube.coord(axis = 'x').bounds[-1,1]).reshape(1,1)
        long_bounds = np.concatenate((long_nums,long_end)).reshape(1+long_size)
        # Convert lat bounds to a 1D array
        lat_size = cube.coord(axis = 'y').bounds.shape[0]
        lat_nums = cube.coord(axis = 'y').bounds[:,0].reshape(lat_size,1)
        lat_end = np.asarray(cube.coord(axis = 'y').bounds[-1,1]).reshape(1,1)
        lat_bounds = np.concatenate((lat_nums,lat_end)).reshape(1+lat_size)
        
        return (np.meshgrid(long_bounds,lat_bounds), 
                cube.coord(axis = 'x').points, cube.coord(axis ='y').points)
    
    @staticmethod
    def _get_horizontal_coord(cube, axis):
        """
        Gets the horizontal coordinate on the supplied cube along the
        specified axis.

        Args:

        * cube:
            An instance of :class:`iris.cube.Cube`.
        * axis:
            Locate coordinates on `cube` along this axis.

        Returns:
            The horizontal coordinate on the specified axis of the supplied
            cube.

        """
        
        coords = cube.coords(axis=axis)
        if len(coords) != 1:
            raise ValueError('Cube {!r} must contain a single 1D {} '
                             'coordinate.'.format(cube.name()), axis)
        return coords[0]

    def __call__(self, src):
        """
        Regrid the supplied :class:`~iris.cube.Cube` on to the target grid of
        this :class:`BackprojectRegridder`.

        Args:

        * src:
            A :class:`~iris.cube.Cube` to be regridded.

        Returns:
            A cube defined with the horizontal dimensions of the target
            and the other dimensions from this cube. The data values of
            this cube will be converted to values on the new grid using
            point-in-cell target backproject regridding.

        """
        # Validity checks.
        if not isinstance(src, iris.cube.Cube):
            raise TypeError("'src' must be a Cube")

        gx = self._get_horizontal_coord(self._src_cube, 'x')
        gy = self._get_horizontal_coord(self._src_cube, 'y')
        src_grid = (gx.copy(), gy.copy())
        sx = self._get_horizontal_coord(src, 'x')
        sy = self._get_horizontal_coord(src, 'y')
        if (sx, sy) != src_grid:
            raise ValueError('The given cube is not defined on the same '
                             'source grid as this regridder.')

        # Call the regridder function.
        res = self.regrid_weighted_backproject_target(src, self.weights,
                                                              self._tgt_cube)
        return res
        
    def regrid_weighted_backproject_target(self, src_cube, weights, grid_cube):
        """
        Return a new cube with the data values calculated using the weighted
        mean of data values from :data:`src_cube` and the weights from
        :data:`weights` regridded onto the horizontal grid of :data:`grid_cube`.

        TODO: Add stuff to this doc
        
        Args:

        * src_cube:
            A :class:`iris.cube.Cube` instance that defines the source
            variable grid to be regridded.
        * weights:
            A :class:`numpy.ndarray` instance that defines the weights
            for the source variable grid cells. Must have the same shape
            as the :data:`src_cube.data`.
        * grid_cube:
            A :class:`iris.cube.Cube` instance that defines the target
            rectilinear grid.

        Returns:
            A :class:`iris.cube.Cube` instance.

        """
        # Get Cartopy CRS instances for the coordinate systems of the
        # source and traget cubes
        src_proj = src_cube.coord_system().as_cartopy_crs()
        tgt_proj = grid_cube.coord_system().as_cartopy_crs()
        
        if src_cube.shape != weights.shape:
            msg = 'The source cube and weights require the same data shape.'
            raise ValueError(msg)

        if src_cube.aux_factories:
            msg = 'All source cube derived coordinates will be ignored.'
            warnings.warn(msg)

        # Get the source cube x and y 2D auxiliary coordinates.
        sx, sy = src_cube.coord(axis='x'), src_cube.coord(axis='y')
        
        if not sx.has_bounds():
            sx.guess_bounds()
            sy.guess_bounds()
        
        # Get the target grid cube x and y dimension coordinates.
        tx, ty = get_xy_dim_coords(grid_cube)
        
        if not tx.has_bounds():
            tx.guess_bounds()
            ty.guess_bounds()

        #TODO Add some checks here so we don't break too easily

        # Create target grid cube x and y cell boundaries.
        tx_depth, ty_depth = tx.points.size, ty.points.size
        tx_dim, = grid_cube.coord_dims(tx)
        ty_dim, = grid_cube.coord_dims(ty)
        
        def squares_from_grid(grid):
            """
            Split the grid into a list of squares where each square is a 5 by 
            2 :class:`numpy.ndarray` of the vertices of the cube in counter
            clockwise orientation starting from the right most vertex
            (which is repeated as the 1st and 5th vertex in the array)
            
            Args:
            
            * grid:
                A grid of points (as would be returned from numpy.meshgrid)
                
            Returns:
                A list of squares
            """
            
            squares = []
            
            for i in range(grid.shape[0]-1):
                for j in range(grid.shape[1]-1):
                
                    square = np.empty((5,2))
                
                    square[0, 0] = grid[i,j,0]
                    square[0, 1] = grid[i,j,1]
                    square[1, 0] = grid[i+1,j,0]
                    square[1, 1] = grid[i+1,j,1]
                    square[2, 0] = grid[i+1,j+1,0]
                    square[2, 1] = grid[i+1,j+1,1]
                    square[3, 0] = grid[i,j+1,0]
                    square[3, 1] = grid[i,j+1,1]
                    square[4, 0] = grid[i,j,0]
                    square[4, 1] = grid[i,j,1]
                    
                    #sometimes needed!                
                    #square = square + [360,0]      
                    
                    squares.append(square)
            
            return squares
        
        def _regrid_indices(t_grid):
            """
            Finds where each source point lies within the target grid

            Args:

            * t_grid:
                The target grid we are regridding onto.

            Returns:
                A tuple of the row and column position of each source data 
                point in the sparse matrix as well as the corresponding 
                ordered data

            """
            
            # row stride length along src_cube data
            row_stride = src_cube.data.shape[1]  
            
            # Row, Column and Data arrays that will be populated and used to 
            # build a sparse matrix
            row = np.empty(src_cube.data.size, dtype = np.int32)
            col = np.empty(src_cube.data.size, dtype = np.int32)
            data = np.empty(src_cube.data.size)
            
            # back projected target grid points
            n_grid = src_proj.transform_points(tgt_proj, t_grid[0], t_grid[1])
            
            # List containing back projected squares (ordered as flat target 
            # array)
            squares = squares_from_grid(n_grid)
            
            # List of indices of source points to check target cell containment
            check_list = np.empty((0,2),dtype = np.int32)
            
            # List of corresponding points in source space
            point = np.empty((0,2))
            
            # Counter for how many source points we have added to sparse matrix
            count = 0
            
            # Loop through each back projected target cell and find which source
            # points are SAFELY within the cell whilst maintaining a list of 
            # points which remain to be checked.
            
            for i ,square in enumerate(squares):
                
                # Get list of indices of each source point contained in the square
                # and those that need checking
                indices, checkcells, l, r = self.get_points_in_square(square)
                
                #populating list of cells to check
                check_list = np.append(check_list,checkcells)
                check_list = check_list.reshape((check_list.size/2,2))
                
                #populating corresponding list of points in space
                temp_point = np.asarray((self.x_points[checkcells[:,1]], 
                                         self.y_points[checkcells[:,0]]))
                temp_point = temp_point.T
                
                point = np.append(point,temp_point)
                point = point.reshape(point.size/2,2)
                
                # If the square contains / intersects some source cells, add
                # the safe cells to the sparse matrix
                
                if len(indices) > 0:
                    
                    # Build a counter that can be used to index columns
                    # (the safe indices are currently listed in a masked array
                    # whose rows correspond to the columns in source grid.
                    # see get_intersected_and_safe_inds)
                    row_vec = np.arange(l,r, dtype = np.int32).reshape((r-l,1))
                    
                    counter = np.zeros(indices.shape, dtype = np.int32)
                    counter += row_vec
                    counter  = ma.masked_where(ma.getmask(indices), counter)

                    compressed = ma.compressed(indices.ravel())
                    
                    #This is the ith target cell in flat index space
                    row[count:count+compressed.size] = i
                    
                    # Calculate the flat source index from the row and
                    # deducing the column from the masked array as explained in
                    # get_intersected_and_safe_inds
                    
                    n_cols = indices.ravel()*row_stride
                    
                    n_cols += (counter).ravel()
                    
                    col[count:count+compressed.size] = n_cols.compressed()
                    
                    # Finally add the corresponding data for the source points
                    # using the masked counter to align with the rows of the 
                    # masked indices array and provide the column index
                    
                    data[count:count+compressed.size] = weights[compressed, ma.compressed(counter)]
                    count += compressed.size
            
            # Magical code to remove any duplicate source cells in our list
            # of cells to check
            
            b = np.ascontiguousarray(check_list)
            b = b.view(np.dtype((np.void, 
                        check_list.dtype.itemsize * check_list.shape[1])))
                        
            _, idx = np.unique(b, return_index=True)
            
            check_list = check_list[idx]
            point = point[idx]
            
            # Forward project the source points into the target space
            trans_points = tgt_proj.transform_points(src_proj, point[:,0], point[:,1])
            
            # Find the x and y indices of the forward projected points in the
            # target grid
            x_indices = np.searchsorted(t_grid[0][0], 
                                        trans_points[:,0], side='right') - 1
            y_indices = np.searchsorted(t_grid[1][:,0], 
                                        trans_points[:,1], side='right') - 1
            
            row_stride_tgt = grid_cube.shape[1]
            
            #Flatten
            
            x_indices = x_indices.reshape(x_indices.size)
            y_indices = y_indices.reshape(y_indices.size)
            
            # Get rid of any invalid indices (cells outside of target grid)
            valid = np.where((x_indices >= 0) & (x_indices < row_stride_tgt) &
                             (y_indices >= 0) & 
                             (y_indices < t_grid[1][:,0].size - 1))
            
            # Get the corresponding indices in flattened source/target
            # grid indices 
            
            flat_inds = y_indices[valid]*row_stride_tgt + x_indices[valid]
            flat_src = check_list[valid,0]*row_stride + check_list[valid,1]
            new_data = weights[check_list[valid,0],check_list[valid,1]]
            
            length = flat_inds.size
            
            # Add checked source points to sparse matrix
            
            row[count:count+length] = flat_inds
            col[count:count+length] = flat_src
            data[count:count+length] = new_data
            
            count += length
            
            return row[:count], col[:count], data[:count]
        
        # Get the indices for which source points are in which target cells
        # and create row, column and data arrays for the sparse matrix
        
        rows, cols, data = _regrid_indices(self.tgt_grid)

        # Now construct a sparse M x N matix, where M is the flattened target
        # space, and N is the flattened source space. The sparse matrix will 
        #then be populated with those source cube points that 
        #contribute to a specific target cube cell.

        # Build our sparse M x N matrix of weights.
        sparse_matrix = csc_matrix((data, (rows, cols)),
                                   shape=(grid_cube.data.size, 
                                          src_cube.data.size))

        # Performing a sparse sum to collapse the matrix to (M, 1).
        sum_weights = sparse_matrix.sum(axis=1).getA()

        # Determine the rows (flattened target indices) that have a
        # contribution from one or more source points.
        rows = np.nonzero(sum_weights)

        # Calculate the numerator of the weighted mean (M, 1).
        numerator = sparse_matrix * src_cube.data.reshape(-1, 1)

        # Calculate the weighted mean payload.
        weighted_mean = ma.masked_all(numerator.shape, dtype=numerator.dtype)
        weighted_mean[rows] = numerator[rows] / sum_weights[rows]

        # Construct the final regridded weighted mean cube.
        dim_coords_and_dims = list(zip((ty.copy(), tx.copy()),
                                       (ty_dim, tx_dim)))
        cube = iris.cube.Cube(weighted_mean.reshape(grid_cube.shape),
                              dim_coords_and_dims=dim_coords_and_dims)
        cube.metadata = copy.deepcopy(src_cube.metadata)

        for coord in src_cube.coords(dimensions=()):
            cube.add_aux_coord(coord.copy())

        return cube
        
    def find_cell(self,pts):
        """
        Finds the index of the cell that contains the given point(s)
        Note: Does not correctly handle points on the boundary of a
        source cell

        Args:

        * pts:
            An array of points in source space

        Returns:
            An array of indices of the cells containing the given points

        """
        # Finds and returns the cell index of the cell that contains the point
        #
        # NOTE: Does not correctly handle points on boundary
    
        index = np.asarray(self.tree.query(pts)[1])

        row_stride = self.src_grid[0].shape[1] -1
        
        # convert flat target space index to grid index
        
        row = (index - index%row_stride)/row_stride
        col = index%row_stride
        
        return np.asarray((row,col)).T
    
    def get_grads(self, square):
        """
        Calculates the gradient and y intercept of the edges of the 
        square in source space.

        Args:

        * square:
            A 5 by 2 :class:`numpy.ndarray` of the vertices of the cube in
            counter clockwise orientation starting from the right most vertex
            (which is repeated as the 1st and 5th vertex in the array)

        Returns:
            A list of (gradient, y intercept) tuples for each edge of
            the given square. Returns (None, None) for vertical lines.

        """
    
        results = np.empty((4,2))

        for i in range(4):
            
            if square[i,0]-square[(i+1)%4,0] != 0:
            
                dy = square[i,1]-square[(i+1)%4,1]    
                dx = square[i,0]-square[(i+1)%4,0]
                gradient = dy/dx
                intercept = square[i,1] - gradient*square[i,0]
            else:
                gradient = intercept = None
            results[i,0] = gradient
            results[i,1] = intercept

        return results

    def get_col_intersections(self, grads, inds, square):
        """
        Finds the exit and entry points of each line of the square through
        columns in the source grid.

        Args:

        * grads:
            A list of (gradient, y intercept) pairs corresponding to each
            edge of the square
        * inds:
            The indices of the source cells containing the corners of the
            square (appropriately ordered)
        * square:
            A 5 by 2 :class:`numpy.ndarray` of the vertices of the cube in
            counter clockwise orientation starting from the right most vertex
            (which is repeated as the 1st and 5th vertex in the array)

        Returns:
            A list of points in source space lying on the midpoint of each 
            column at the entry and exit points of each edge going through
            the column (ordered lower point then higher point)

        """

        #list of upper and lower exit points from each column at 
        #(x_value_at_midpoint_of_column, y_intercept)
        intersections = []

        
        # Populates the list by finding the 
        
        for i, line in enumerate(grads):

            start_column = inds[i][1]
            start_y = square[i][1]
            end_column = inds[i+1][1]
            end_y = square[i+1][1]

            low_lim = int(min(start_column, end_column))
            up_lim = int(max(start_column, end_column))
            
            #Starting column
    
            start_x = self.src_grid[0][0,start_column]
            end_x = self.src_grid[0][0,start_column+1]
            
            midpoint = (start_x + end_x)/2
            
            if start_column < end_column:
            
                exit = end_x*line[0] + line[1]
                upper = max(exit,start_y)
                lower = min(exit,start_y)
                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))
            elif start_column == end_column:
                
                upper = max(end_y,start_y)
                lower = min(end_y,start_y)
                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))
            else:
                
                exit = start_x*line[0] + line[1]
                upper = max(exit,start_y)
                lower = min(exit,start_y)
                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))

            #Ending column
            start_x = self.src_grid[0][0, end_column]
            end_x = self.src_grid[0][0, end_column+1]
            
            midpoint = (start_x + end_x)/2
            
            if start_column > end_column:
                
                exit = end_x*line[0] + line[1]
                upper = max(exit,end_y)
                lower = min(exit,end_y)
                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))
            elif start_column != end_column:
                
                exit = start_x*line[0] + line[1]
                upper = max(exit,end_y)
                lower = min(exit,end_y)
                
                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))
            #Intermediate columns
            for index in range(low_lim+1, up_lim):
        
                start_x = self.src_grid[0][0,index]
                end_x = self.src_grid[0][0,index+1]
                
                midpoint = (start_x + end_x)/2
            
                upper = end_x*line[0] + line[1]
                lower = start_x*line[0] + line[1]

                intersections.append((midpoint, lower))
                intersections.append((midpoint, upper))

        return intersections

    def intersected_and_safe_inds(self, square):
        """
        Gets the coordinates in index space of the source points certainly
        contained within and possibly contained within the given square

        Args:

        * square:
            A 5 by 2 :class:`numpy.ndarray` of the vertices of the cube in
            counter clockwise orientation starting from the right most vertex
            (which is repeated as the 1st and 5th vertex in the array)

        Returns:
            A tuple of safe cells (definitely contained in the square) and
            intersected cells (lying in a cell intersected by an edge) which
            need further checking. Both are arrays of coordinates in index
            space. Also returns indices of the left and right bounding 
            columns.
        """
    
        grads = self.get_grads(square)
        
        #find the cells containing the corners of the square
        inds = self.find_cell(square)
        
        #get exit and entry points for each column
        intersections = self.get_col_intersections(grads, inds, square)
        
        #find the exit and entry points in index space
        bounding_inds = self.find_cell(intersections)
        
        intersected_inds = []
        safe_inds = []
        
        #populate a list of indices of all cells intersected by lines
        for i in range(int(len(bounding_inds)/2)):
        
            start_ind = bounding_inds[2*i]
            end_ind = bounding_inds[2*i+1]
            
            intersected_inds.append(start_ind)
            intersected_inds.append(end_ind)
            
            start = int(min(start_ind[0], end_ind[0]))
            end = int(max(start_ind[0], end_ind[0]))
            # Add all source cells between start and end on this column
            # i.e. all source cells intersected by the line in this column
            #Could be quicker?? Vectorise?
            for j in range(start, end):
                intersected_inds.append((j,start_ind[1]))
        
        
        if len(intersected_inds) == 0:
            return None
            
        intersected_inds = np.array(intersected_inds, dtype = np.int32)
        
        # populate a list of indices of all cells fully contained between the lines
        # by, for each column, finding the range between the lower and upper 
        # intersections in the column and adding this to a list. The ith element
        # of the list corresponds to the range on the ith column
        
        cols = [index[1] for index in intersected_inds]
        
        #bounding columns
        left_col = int(min(cols))
        right_col = int(max(cols))
        
        # Calculates the largest range within a column. Used to build masked array
        diff = 0
        for i in range(left_col, right_col + 1):
            indices, = np.where(intersected_inds[:,1] == i)
            rows = intersected_inds[indices,0]
            
            if len(rows) == 4:
                lowest_row = int(min(rows))
                highest_row = int(max(rows))
                if diff < highest_row - lowest_row - 1:
                    diff = highest_row - lowest_row - 1
                    
                safe_inds.append(np.arange(lowest_row + 1, highest_row, dtype=np.int32))
            else:
                safe_inds.append(np.empty(0))
        
        # Create a masked array to store each column range
        marr = ma.masked_all((len(safe_inds), diff), dtype = np.int32)

        for i in range(len(safe_inds)):
            marr[i,:safe_inds[i].size] = safe_inds[i]
            
        safe_inds = marr
        
        return intersected_inds, safe_inds, left_col, right_col + 1
               
    def get_points_in_square(self, square):
        """
        See intersected_and_safe_inds.

        """
        
        check_cells, safe_cells, l, r = self.intersected_and_safe_inds(square) 
                
        return (safe_cells,check_cells, l, r)

class PointInCell(object):
    """
    This class describes the point-in-cell regridding scheme for regridding
    over one or more orthogonal coordinates, typically for use with
    :meth:`iris.cube.Cube.regrid()`.

    """
    def __init__(self, weights):
        """
        Point-in-cell regridding scheme suitable for regridding over one
        or more orthogonal coordinates.

        Args:

        * weights:
            A :class:`numpy.ndarray` instance that defines the weights
            for the grid cells of the source grid. Must have the same shape
            as the data of the source grid.

        """
        self.weights = weights

    def regridder(self, src_grid, target_grid):
        """
        Creates a point-in-cell target grid backproject regridder to perform
        regridding from the source grid to the target grid.

        Typically you should use :meth:`iris.cube.Cube.regrid` for
        regridding a cube. There are, however, some situations when
        constructing your own regridder is preferable. These are detailed in
        the :ref:`user guide <caching_a_regridder>`.

        Args:

        * src_grid:
            The :class:`~iris.cube.Cube` defining the source grid.
        * target_grid:
            The :class:`~iris.cube.Cube` defining the target grid.

        Returns:
            A callable with the interface:

                `callable(cube)`

            where `cube` is a cube with the same grid as `src_grid`
            that is to be regridded to the `target_grid`.

        """
        return BackprojectRegridder(src_grid, target_grid, self.weights)
