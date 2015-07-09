import iris
import numpy as np

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
