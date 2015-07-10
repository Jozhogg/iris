import iris
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import iris.plot as iplot
import iris.quickplot as qplot
from cell_intersects import CellIntersects
from cube_grid import cube_grid

latitude = iris.coords.DimCoord(range(50), standard_name='latitude', units='degrees')
longitude = iris.coords.DimCoord(range(50), standard_name='longitude', units='degrees')
data = np.random.random([50,50])

cube = iris.cube.Cube(data,standard_name='air_temperature',units = 'kelvin',
dim_coords_and_dims = [(latitude, 0), (longitude, 1)])

grid = cube_grid(cube)

Cell = CellIntersects(grid)

square = Cell.gen_random_square()

check_cells, safe_cells = Cell.intersected_and_safe_inds(square)

lat_points = cube.coord('latitude').points
long_points = cube.coord('longitude').points

cells_in_square = safe_cells.tolist()

    
#hgsfhsgfsf


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

for index in check_cells:
    point = lat_points[index[1]],long_points[index[0]]
    plt.plot(point[0],point[1],'bo')
    if is_in_square(square,point):
        cells_in_square.append(index)

testpoint = np.array([25,25])

#print is_in_square(square, testpoint)

for index in cells_in_square:
    #plt.plot(cell[1]+0.5,cell[0]+0.5,'ro',markersize = 10)
    plt.plot(grid[0][index[0], index[1]]+0.5,grid[1][index[0], index[1]]+0.5, 'ro', markersize = 13)
plt.plot(square[:,0],square[:,1])
    
plt.plot(cube_grid(cube)[0],cube_grid(cube)[1],'go',markersize = 3)
plt.show()
