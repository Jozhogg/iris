import numpy as np
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import cartopy.crs as ccrs
from scipy import spatial
import matplotlib.pyplot as plt
from regrid import PointInCell
from regridder import Regridder

np.set_printoptions(threshold = np.nan)

src = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/35_35_source.nc")
tgt = iris.load_cube("/project/atk/_tmp_data_store/avd_sprint/ite_veg_fraction/ukv_grid/35_35_target.nc")

print src.coords(axis='x')
arr = np.arange(338436).reshape((1422, 238))
src.data = arr

iplt.contourf(src, 1000)

rg_src = src.regrid(tgt, PointInCell(np.ones(338436).reshape((1422, 238))))

lin_rg = src.regrid(tgt, iris.analysis.Linear(extrapolation_mode = "mask"))


print rg_src
print rg_src.data
print lin_rg
print lin_rg.data

ax = plt.gca()
ax.coastlines()

plt.show()

