import iris
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import iris.plot as iplot
import iris.quickplot as qplot
from scipy import spatial

class Regridder():

    def cube_grid(self, cube):
        # Creates and returns a numpy meshgrid of lats and longs from cube
        
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
        
        return np.meshgrid(long_bounds,lat_bounds), cube.coord(axis = 'x').points, cube.coord(axis ='y').points

    def __init__(self, src_cube, tgt_cube):
    
        # Instantiate a regridder for the given source and target cubes
        
        
        # Get the source cube's cell grid
        self.src_grid, self.x_points, self. y_points = self.cube_grid(src_cube)
        
        self.tgt_grid = self.cube_grid(tgt_cube)[0]
        
        # calculate x and y midpoints of source cell grid
        cellx = np.zeros((self.src_grid[0].shape[0]-1, 
                                    self.src_grid[0].shape[1]-1))
        celly = np.zeros((self.src_grid[0].shape[0]-1, 
                                    self.src_grid[0].shape[1]-1))

        for i in range(self.src_grid[0].shape[1]-1):
            cellx[:,i] = float(self.src_grid[0][0,i] + self.src_grid[0][0,i+1])/2
        for i in range(self.src_grid[0].shape[0]-1):
            celly[i,:] = float(self.src_grid[1][i,0] + self.src_grid[1][i+1,0])/2
        
        #Create KDTree of centre points of cells
        self.tree = spatial.KDTree(zip(cellx.ravel(), celly.ravel()))

    def find_cell(self,pts):
    
        # Finds and returns the cell index of the cell that contains the point
        #
        # NOTE: Does not correctly handle points on boundary
    
        index = np.asarray(self.tree.query(pts)[1])

        row_stride = self.src_grid[0].shape[1] -1
    
        row = (index - index%row_stride)/row_stride
        col = index%row_stride
    
        return np.asarray((row,col)).T

    def get_grads(self, square):
    
        # Return a list of (gradient, y intercept) tuples for each edge of
        # the given square. Returns (None, None) for vertical line
    
        results = []

        for i in range(4):
            
            if square[i,0]-square[(i+1)%4,0] != 0:
                gradient = (square[i,1]-square[(i+1)%4,1])/(square[i,0]-square[(i+1)%4,0])
                intercept = square[i,1] - gradient*square[i,0]
            else:
                gradient = intercept = None
            results.append((gradient,intercept))

        return np.asarray(results)

    def get_col_intersections(self, grads, inds, square):

        # Returns a list of positions of entry and exit points for lines of the square through each column
        # in order lower, upper

        #list of upper and lower exit points from each column at (x_value_at_start_of_column, y_intercept)
        intersections = []
    
        for i, line in enumerate(grads):

            start_column = inds[i][1]
            start_y = square[i][1]
            end_column = inds[i+1][1]
            end_y = square[i+1][1]

            low_lim = min(start_column, end_column)
            up_lim = max(start_column, end_column) 
            
            #Starting cell
            
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

            #Ending cell
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
            #Intermediate cells
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

            if(start_ind[1] != end_ind[1]):
                print("COLUMNS MISMATCH:\n")
                print start_ind, end_ind
            
            intersected_inds.append(start_ind)
            intersected_inds.append(end_ind)

            start = min(start_ind[0], end_ind[0])
            end = max(start_ind[0], end_ind[0])
            
            for j in range(start, end):
                intersected_inds.append((j,start_ind[1]))
        
        if len(intersected_inds) == 0: 
        
            return None    
            
        intersected_inds = np.array(intersected_inds)
        
        #populate a list of indices of all cells fully contained between the lines
        left_col = min([index[1] for index in intersected_inds])
        right_col = max([index[1] for index in intersected_inds])

        for i in range(left_col, right_col + 1):

            indices, = np.where(intersected_inds[:,1] == i)
            if len(intersected_inds[indices,0]) > 0:
                lowest_row = min(intersected_inds[indices,0])
                highest_row = max(intersected_inds[indices,0])

                for j in range(lowest_row, highest_row):
                    if not j in intersected_inds[indices,0]:
                        safe_inds.append((j,i))
    
        return np.asarray(intersected_inds), np.asarray(safe_inds)
    
    def is_in_square(self,square,point):
        grads = self.get_grads(square)
        position = []    
        #vertical
        for i, line in enumerate(grads):

            start_x = min(square[i][0],square[i+1][0])
            end_x = max(square[i][0],square[i+1][0])
            if start_x < point[0] < end_x:
                y = point[0]*line[0] + line[1]
                if point[1] > y: 
                    position.append(True)
                if point[1] < y:
                    position.append(False)               
        if len(position) != 2:
            pass
            #print 'position is wrong size!!!  ' + str(len(position))
        else:
            if not position[0] == position[1]:
                return True
            else:
                return False
                
    def get_points_in_square(self, square):
    
        #returns the indices of points in the given square

        check_cells, safe_cells = self.intersected_and_safe_inds(square)
       
        
        cells_in_square = safe_cells.tolist()
        
        for index in check_cells:
            point = self.x_points[index[1]], self.y_points[index[0]]

            if self.is_in_square(square,point):
                cells_in_square.append(index)
                
        return np.array(cells_in_square)


