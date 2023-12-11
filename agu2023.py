# %%
import numpy as np
import xarray as xr
import os

# Import supporting module
import sys

scriptsDir = os.path.dirname(__file__)
sys.path.append(scriptsDir)
import agu2023mod as agu23

# Read configuration
config = agu23.read_config()
for option, value in config.items("DEFAULT"):
    print(f"DEFAULT: {option} = {value}")
for section in config.sections():
    for option in config[section]:
        print(f"{section}: {option} = {config[section][option]}")
this_dir = config["runs"]["this_dir"]
expt_list = config.getlist("runs", "expt_list")

# Import general CTSM Python utilities
sys.path.append(config["system"]["my_ctsm_python_gallery"])
import utils


# %% Import

os.chdir(this_dir)

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

agu23.make_ts_plot(
    expt_list,
    ds0,
    var_list,
    abs_diff=abs_diff,
    rel_diff=rel_diff,
    y2y_diff=y2y_diff,
    cropland_only=cropland_only,
    rolling=rolling,
)
agu23.process_and_make_plot(
    expt_list,
    ds0,
    var_list,
    abs_diff=abs_diff,
    rel_diff=rel_diff,
    y2y_diff=y2y_diff,
    cropland_only=cropland_only,
    rolling=rolling,
)


# %% h2 files
abs_diff = False
rel_diff = False
y2y_diff = False
cropland_only = True
# var_list = ["TOTLITC", "TOTLITC_1m", "TOTLITN", "TOTLITN_1m", "TOTSOMC", "TOTSOMC_1m", "TOTSOMN", "TOTSOMN_1m"]
var_list = ["TOTSOMC"]

agu23.process_and_make_plot(
    expt_list,
    ds2,
    var_list,
    abs_diff=abs_diff,
    rel_diff=rel_diff,
    y2y_diff=y2y_diff,
    cropland_only=cropland_only,
)


# %% h3 files
abs_diff = False
rel_diff = False
y2y_diff = False
# var_list = ["CROPPROD1C", "CROPPROD1C_LOSS"]
var_list = ["CROPPROD1C_LOSS"]

agu23.process_and_make_plot(
    expt_list,
    ds3,
    var_list,
    abs_diff=abs_diff,
    rel_diff=rel_diff,
    y2y_diff=y2y_diff,
    cropland_only=False,
)


# %% GRAINC_TO_FOOD_ANN
abs_diff = False
rel_diff = False
y2y_diff = False

agu23.process_and_make_plot(
    expt_list,
    ds1,
    "GRAINC_TO_FOOD_ANN",
    abs_diff=abs_diff,
    rel_diff=rel_diff,
    y2y_diff=y2y_diff,
    cropland_only=False,
)


# %% What fraction of each gridcell is cropland?

da = ds2[0]["land1d_wtgcell"].where(ds2[0]["land1d_ityplunit"] == 2, drop=True)
da = da.rename({"landunit": "gridcell"})
ds2[0]["tmp"] = da
da = utils.grid_one_variable(ds2[0], "tmp")
da.plot()
