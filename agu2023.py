# %%
import numpy as np
import xarray as xr
import os

import sys
sys.path.append(os.path.dirname(__file__))
import agu2023mod as agu23

# Import general CTSM Python utilities
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils



# %% Import

os.chdir("/Users/Shared/CESM_runs/agu2023_10x15v3.3")

# expt_list = ["Toff_Roff", "Thi_Rhi", "Toff_Roff_fromHi", "Thi_Rhi_fromOff", "Thi_Roff_fromOff", "Toff_Rhi_fromHi"]
# expt_list = ["Toff_Roff", "Thi_Rhi_fromOff"]
# expt_list = ["Thi_Rhi", "Toff_Roff_fromHi"]
# expt_list = ["Toff_Roff", "Thi_Rhi_fromOff", "Toff_Rhi_fromOff"]
# expt_list = ["Thi_Rhi", "Toff_Roff_fromHi", "Thi_Roff_fromHi"]
# expt_list = ["Toff_Roff", "Thi_Rhi_fromOff", "Thi_Roff_fromOff"]
# expt_list = ["Thi_Rhi", "Toff_Roff_fromHi", "Toff_Rhi_fromHi"]
expt_list = ["Toff_Roff", "Thi_Rhi", "Thi_Rhi_fromOff", "Thi_Roff_fromOff"]

ds0 = []
ds1 = []
ds2 = []
ds3 = []
for e, expt_name in enumerate(expt_list):
    ds0.append(xr.open_dataset(agu23.get_file(0, expt_name)))
    ds1.append(xr.open_dataset(agu23.get_file(1, expt_name)))
    ds2.append(xr.open_dataset(agu23.get_file(2, expt_name)))
    ds3.append(xr.open_dataset(agu23.get_file(3, expt_name)))


# %% h0 files
abs_diff = False
rel_diff = False
y2y_diff = False
cropland_only = False
rolling = None
# var_list = ["SOILC_HR", "NEP", "NEE", "NBP"]
var_list = ["NBP"]

agu23.make_plot(expt_list, ds0, var_list, abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=cropland_only, rolling=rolling)


# %% h2 files
abs_diff = False
rel_diff = False
y2y_diff = False
cropland_only = True
# var_list = ["TOTLITC", "TOTLITC_1m", "TOTLITN", "TOTLITN_1m", "TOTSOMC", "TOTSOMC_1m", "TOTSOMN", "TOTSOMN_1m"]
var_list = ["TOTSOMC"]

agu23.make_plot(expt_list, ds2, var_list, abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=cropland_only)


# %% h3 files
abs_diff = False
rel_diff = False
y2y_diff = False
# var_list = ["CROPPROD1C", "CROPPROD1C_LOSS"]
var_list = ["CROPPROD1C_LOSS"]

agu23.make_plot(expt_list, ds3, var_list, abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=False)


# %% GRAINC_TO_FOOD_ANN
abs_diff = False
rel_diff = False
y2y_diff = False

agu23.make_plot(expt_list, ds1, "GRAINC_TO_FOOD_ANN", abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=False)



# %% What fraction of each gridcell is cropland?

da = ds2[0]["land1d_wtgcell"].where(ds2[0]["land1d_ityplunit"] == 2, drop=True)
da = da.rename({"landunit": "gridcell"})
ds2[0]["tmp"] = da
da = utils.grid_one_variable(ds2[0], "tmp")
da.plot()


# %%

a = xr.open_dataset("/Users/Shared/CESM_runs/agu2023_10x15/agu2023_10x15_fut_Toff_Roff_fromHi.clm2.h1.2095-01-01-00000.nc")
b = xr.open_dataset("/Users/Shared/CESM_runs/agu2023_10x15/agu2023_10x15_fut_Toff_Roff_fromHi.clm2.h1.2096-01-01-00000.nc")

thisVar = "land1d_wtgcell"
np.array_equal(a[thisVar].values, b[thisVar].values, equal_nan=True)