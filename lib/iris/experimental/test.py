import numpy as np
import iris
import iris.plot as iplt
import iris.quickplot as qplt
#import cartopy.crs as ccrs
#from scipy import spatial

import matplotlib.pyplot as plt

from new_regrid import PointInCell


#import iris.tests.stock as stock

np.set_printoptions(threshold = np.nan)


#src = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/35_35_source.nc")
#tgt = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/35_35_target.nc")
'''
sample_dir = "/project/avd/iris/resources/git/iris-sample-data/sample_data/"

iris.config.SAMPLE_DATA_DIR = sample_dir

fname1 = iris.sample_data_path("pre-industrial.pp")
fname2 = iris.sample_data_path("uk_hires.pp")

tgt = iris.load_cube(fname1)
src = iris.load_cube(fname2, "air_potential_temperature")
src = src[0,0,:,:]


tgt = tgt.intersection(latitude  = (53,55))
tgt = tgt.intersection(longitude  = (-8,-2))

iplt.contourf(src)


rg_src = src.regrid(tgt, PointInCell(np.ones(src.data.size).reshape(src.data.shape)))

'''
numb = '30'
src = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/"+numb+"_"+numb+"_source.nc")
tgt = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/"+numb+"_"+numb+"_target.nc") 

#print src.coords(axis='x')
#arrx = np.arange(src.data.shape[0])
#arry = np.arange(src.data.shape[1])
#grid = np.meshgrid(arrx,arry)
"""latitude = iris.coords.DimCoord(range(50), standard_name='latitude', units='degrees')
longitude = iris.coords.DimCoord(range(50), standard_name='longitude', units='degrees')
data = np.random.random([50,50])
src = iris.cube.Cube(data,standard_name='air_temperature',units = 'kelvin',
dim_coords_and_dims = [(latitude, 0), (longitude, 1)])
"""
data = np.fromfunction(lambda i,j: np.sin((i*3.14)/180)*np.cos((j*3.14)/180), src.data.shape)

#data = np.zeros(src.data.size).reshape(src.data.shape)
#val =  data.shape[0]/10
#for i in range(10):
#    data[i*val:i*val+val/2, :] = 10    

src.data = data
"""
latitude = iris.coords.DimCoord(range(15), standard_name='latitude', units='degrees')
longitude = iris.coords.DimCoord(range(15), standard_name='longitude', units='degrees')
data = np.random.random([15,15])
tgt = iris.cube.Cube(data,standard_name='air_temperature',units = 'kelvin',
dim_coords_and_dims = [(latitude, 0), (longitude, 1)])
"""
#print src.data.shape
#print src.data.size

plt.subplot(1,2,1)
qplt.contourf(src)



rg_src = src.regrid(tgt, PointInCell(np.ones(src.data.size).reshape((src.data.shape))))

plt.subplot(1,2,2)
qplt.contourf(rg_src)

#lin_rg = src.regrid(tgt, iris.analysis.Linear(extrapolation_mode = "mask"))


#print rg_src
#print rg_src.data
#print lin_rg
#print lin_rg.data

ax = plt.gca()
ax.coastlines()

plt.show()

