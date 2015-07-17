import numpy as np
import iris
#import iris.plot as iplt
#import iris.quickplot as qplt
#import cartopy.crs as ccrs
#from scipy import spatial

#import matplotlib.pyplot as plt

from new_regrid import PointInCell
from regridder import Regridder

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
numb = '10'
src = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/"+numb+"_"+numb+"_source.nc")
tgt = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/"+numb+"_"+numb+"_target.nc") 

#print src.coords(axis='x')
arr = np.arange(src.data.size).reshape((src.data.shape))
src.data = arr

#print src.data.shape
#print src.data.size

#iplt.contourf(src, 10)

"""src = stock.lat_lon_cube()
src.coord(axis='x').guess_bounds()
src.coord(axis='y').guess_bounds()

tgt = src.copy()
tgt.coord(axis='x').coord_system = stock.RotatedGeogCS(30, 30)
tgt.coord(axis='y').coord_system = stock.RotatedGeogCS(30, 30)"""

rg_src = src.regrid(tgt, PointInCell(np.ones(src.data.size).reshape((src.data.shape))))


#lin_rg = src.regrid(tgt, iris.analysis.Linear(extrapolation_mode = "mask"))


print rg_src
print rg_src.data
#print lin_rg
#print lin_rg.data

#ax = plt.gca()
#ax.coastlines()

#plt.show()

