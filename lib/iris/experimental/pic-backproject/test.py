import numpy as np
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import cartopy.crs as ccrs
from scipy import spatial
import matplotlib.pyplot as plt
from regridder import Regridder

sample_dir = "/project/avd/iris/resources/git/iris-sample-data/sample_data/"

iris.config.SAMPLE_DATA_DIR = sample_dir

fname1 = iris.sample_data_path("pre-industrial.pp")
fname2 = iris.sample_data_path("uk_hires.pp")

src = iris.load_cube(fname1)
tgt = iris.load_cube(fname2, "air_potential_temperature")

src = src.intersection(latitude  = (53,55))
src = src.intersection(longitude  = (-8,-2))

gridder = Regridder(tgt, src)

grid = gridder.tgt_grid

n_grid = tgt.coord_system().as_cartopy_crs().transform_points(src.coord_system().as_cartopy_crs(), grid[0], grid[1])

print n_grid.shape

n_grid = n_grid[:,:,0:2]

print n_grid[0].shape

squares = []

for i in range(n_grid.shape[0]-1):
    for j in range(n_grid.shape[1]-1):
    
        square = []
    
        square.append((n_grid[i,j,0], n_grid[i,j,1]))
        square.append((n_grid[i+1,j,0], n_grid[i+1,j,1]))
        square.append((n_grid[i+1,j+1,0], n_grid[i+1,j+1,1]))
        square.append((n_grid[i,j+1,0], n_grid[i,j+1,1]))
        square.append((n_grid[i,j,0], n_grid[i,j,1]))        
        
        squares.append(np.array(square))



qplt.contourf(tgt[0,0,:,:])

#plt.plot(n_grid[0,:,0], n_grid[0,:,1], 'ro', transform = src.coord_system().as_cartopy_crs())
#plt.plot(n_grid[10,:,0], n_grid[10,:,1], 'bo', transform = src.coord_system().as_cartopy_crs())


"""for i, square in enumerate(squares):
    plt.plot(square[:,0],square[:,1], 'ro', transform = tgt.coord_system().as_cartopy_crs())
    indices = gridder.get_points_in_square(square)
    if len(indices) > 0:
        total = 0
        for index in indices:
            total += tgt.data[0,0,index[0], index[1]]
        
        average = total/len(indices)
        
        row_stride = src.shape[0]
        print row_stride
        
        row = (i - i%row_stride)/row_stride -1
        col = i%row_stride
        
        print row, col
        
        src.data[row,col] = average
    else:
        row_stride = src.shape[0]
        print row_stride
        row = (i - i%row_stride)/row_stride
        col = i%row_stride
        
        print row, col
        
        src.data[col,row] = -1"""
        
        
print squares[2]

squares[2] = squares[2] + [360,0]

plt.plot(squares[2][:,0],squares[2][:,1],'ro', transform = tgt.coord_system().as_cartopy_crs())
plt.plot(squares[2][:,0],squares[2][:,1], transform = tgt.coord_system().as_cartopy_crs())

indices = gridder.get_points_in_square(squares[2])



qplt.contourf(src)

print src.data

#print indices

grid = np.meshgrid(gridder.x_points, gridder.y_points)

plt.plot(grid[0].ravel(), grid[1].ravel() , 'bo', transform = tgt.coord_system().as_cartopy_crs(), markersize=1)

#plt.plot(squares[20][:,0], squares[20][:,1])
##plt.plot(squares[20][:,0], squares[20][:,1], 'ro')
for index in indices[::5]:
    plt.plot(gridder.x_points[index[1]], gridder.y_points[index[0]], transform = tgt.coord_system().as_cartopy_crs())

ax = plt.gca()
ax.coastlines()


plt.show()
