import iris
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import iris.plot as iplot
import iris.quickplot as qplot
from cell_intersects import CellIntersects
from cube_grid import cube_grid
from scipy import spatial

class Regridder():

    def __init__(self, grid):

        self.grid = grid
        self.g_size = len(grid[0])

        cellx = celly = np.zeros((grid[0].shape[0]-1, grid[0].shape[1]-1))

        for i in range(grid[0].shape[1]-1):
            cellx[:,i] = float(grid[0][0,i] + grid[0][0,i+1])/2
        for i in range(grid[0].shape[0]-1):
            celly[i,:] = float(grid[1][i,0] + grid[1][i+1,0])/2
    
        celly = celly.T
            
        self.tree = spatial.KDTree(zip(cellx.ravel(), celly.ravel()))
     

    def gen_random_square(self):
          
        # Returns a random convex square in the grid of size (g_size,g_size)
        # with points in ordering of the convex hull (Assumes grid is square 
        # of size g_size

        square = np.random.random((4,2))*(self.g_size - 1)

        #find left most point
        x_ord_ind = np.argsort(square[:,0])

        inds = [0,1,2,3]
        pts = []

        pts.append(square[x_ord_ind[0], :])

        last_angle = -math.pi
        convex = True

        while len(inds) > 0:
                
            if not x_ord_ind[0] in inds:
                convex = False
                break

            cur_angle = math.pi
            cur_index = -1
            for index in inds:
            
                opp = -pts[-1][1] + square[index,1]
                adj = -pts[-1][0] + square[index,0]
                angle = math.atan2(opp,adj)
                if math.isnan(angle):
                    pass
                elif angle < cur_angle:
                    cur_index = index
                    cur_angle = angle
                
                if( (cur_angle-last_angle)%math.pi < 0):
                    convex = False
            
            pts.append(square[cur_index,:])
            inds.remove(cur_index)
            last_angle = cur_angle
        
        if convex:

            pts = np.array(pts)
            return pts
        else:
            return self.gen_random_square()

    def find_cell(self,pts):
    
        #Finds and returns the cell index of the cell that contains the point
        #
        # NOTE: Does not correctly handle points on boundary
    
        index = np.asarray(self.tree.query(pts)[1])

        row_stride = self.grid[0].shape[1] -1
    
        row = (index - index%row_stride)/row_stride
        col = index%row_stride
    
        return np.asarray((col,row)).T    

    def get_grads(self, square):
    
        results = []

        for i in range(4):
            
            gradient = (square[i,1]-square[(i+1)%4,1])/(square[i,0]-square[(i+1)%4,0])
            intercept = square[i,1] - gradient*square[i,0]
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
            if start_column < end_column:
                end_x = self.grid[0][0,start_column+1]
                exit = end_x*line[0] + line[1]
                upper = max(exit,start_y)
                lower = min(exit,start_y)
                intersections.append((end_x -0.1, lower))
                intersections.append((end_x -0.1, upper))
            elif start_column == end_column:
                start_x = self.grid[0][0,start_column]
                upper = max(end_y,start_y)
                lower = min(end_y,start_y)
                intersections.append((start_x +0.1, lower))
                intersections.append((start_x +0.1, upper))
            else:
                start_x = self.grid[0][0,start_column]
                exit = start_x*line[0] + line[1]
                upper = max(exit,start_y)
                lower = min(exit,start_y)
                intersections.append((start_x +0.1, lower))
                intersections.append((start_x +0.1, upper))

            #Ending cell
            if start_column > end_column:
                end_x = self.grid[0][0,end_column+1]
                exit = end_x*line[0] + line[1]
                upper = max(exit,end_y)
                lower = min(exit,end_y)
                intersections.append((end_x -0.1, lower))
                intersections.append((end_x -0.1, upper))
            elif start_column != end_column:
                start_x = self.grid[0][0,end_column]
                exit = start_x*line[0] + line[1]
                upper = max(exit,end_y)
                lower = min(exit,end_y)
                intersections.append((start_x +0.1, lower))
                intersections.append((start_x +0.1, upper))
            #Intermediate cells
            for index in range(low_lim+1, up_lim):
        
                start_x = self.grid[0][0,index]
                end_x = self.grid[0][0,index+1]
            
                upper = end_x*line[0] + line[1]
                lower = start_x*line[0] + line[1]

                # NOTE:added 0.1 to move into cell, won't work in general! 

                intersections.append((start_x + 0.1, lower))
                intersections.append((end_x - 0.1, upper))

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

            intersected_inds.append(start_ind)
            intersected_inds.append(end_ind)
            start = min(start_ind[0], end_ind[0])
            end = max(start_ind[0], end_ind[0])
            
            for j in range(start, end):
                intersected_inds.append((j,start_ind[1]))
    
        intersected_inds = np.array(intersected_inds)

        #populate a list of indices of all cells fully contained between the lines
        left_col = min([index[1] for index in intersected_inds])
        right_col = max([index[1] for index in intersected_inds])

        for i in range(left_col, right_col + 1):

            indices, = np.where(intersected_inds[:,1] == i)
            
            lowest_row = min(intersected_inds[indices,0])
            highest_row = max(intersected_inds[indices,0])

            for j in range(lowest_row, highest_row):
                if not j in intersected_inds[indices,0]:
                    safe_inds.append((j,i))
    
        return np.asarray(intersected_inds), np.asarray(safe_inds)
    
    def cube_grid(cube):
        #creates and returns a numpy meshgrid of lats and longs from cube
        #guess bounds if necessary
        if cube.coord('latitude').bounds == None:
            cube.coord('latitude').guess_bounds()
        if cube.coord('longitude').bounds == None:
            cube.coord('longitude').guess_bounds()
        #convert long bounds to a 1D array
        long_size = cube.coord('longitude').bounds.shape[0]  
        long_nums = cube.coord('longitude').bounds[:,0].reshape(long_size,1)
        long_end = np.asarray(cube.coord('longitude').bounds[-1,1]).reshape(1,1)
        long_bounds = np.concatenate((long_nums,long_end)).reshape(1+long_size)
        #convert lat bounds to a 1D array
        lat_size = cube.coord('latitude').bounds.shape[0]  
        lat_nums = cube.coord('latitude').bounds[:,0].reshape(lat_size,1)
        lat_end = np.asarray(cube.coord('latitude').bounds[-1,1]).reshape(1,1)
        lat_bounds = np.concatenate((lat_nums,lat_end)).reshape(1+lat_size)
        
        return np.meshgrid(lat_bounds,long_bounds)
    
    def is_in_square(square,point):
        grads = Cell.get_grads(square)
        position = []    
        #vertical
        for i, line in enumerate(grads):
            print i
            start_x = min(square[i][0],square[i+1][0])
            end_x = max(square[i][0],square[i+1][0])
            print start_x, end_x
            if start_x < point[0] < end_x:
                y = point[0]*line[0] + line[1]
                if point[1] > y: 
                    position.append(True)
                if point[1] < y:
                    position.append(False)
        print position                
        if len(position) != 2:
            print 'position is wrong size!!!  ' + str(len(position))
        else:
            if not position[0] == position[1]:
                return True
            else:
                return False
                
    def get_cells_in_square(square):
        
        grid = cube_grid(cube)

        Cell = CellIntersects(grid)

        square = Cell.gen_random_square()

        check_cells, safe_cells = Cell.intersected_and_safe_inds(square)

        lat_points = cube.coord('latitude').points
        long_points = cube.coord('longitude').points
        
        cells_in_square = safe_cells.tolist()
        
        for index in check_cells:
            point = lat_points[index[1]],long_points[index[0]]

            if is_in_square(square,point):
                cells_in_square.append(index)
                
        return cells_in_square


