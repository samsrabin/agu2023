# %%
import xarray as xr
import os
import pickle
import glob

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
os.chdir(this_dir)


# %% Define

def get_das(expt_list, var):
    pattern_yearrange = "[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]"
    das = []
    for expt in expt_list:
        pattern = f"*{expt}.clm2.{pattern_yearrange}.pickle"
        file_in = glob.glob(pattern)
        if len(file_in) > 1:
            raise RuntimeError(f"Found {len(file_in)} matches: {file_in}")
        elif len(file_in) == 0:
            raise RuntimeError(f"Found 0 matches for {pattern}")

        with open(file_in[0], "rb") as handle:
            dict_in = pickle.load(handle)
        da = dict_in[var]

        # Ignore extra years
        da = da.sel(time=slice("1901-01-01", "2100-12-31"))

        das.append(da)

    return das


# %%
import importlib
importlib.reload(agu23)

expt_list = ["Toff_Roff", "Thi_Rhi", "Thi_Rhi_fromOff", "Thi_Roff_fromOff"]
var = "NBP"
abs_diff = False
rel_diff = False
y2y_diff = False
cropland_only = False
rolling = None
do_cumsum = True

das = get_das(expt_list, var)
agu23.make_plot(expt_list, abs_diff, rel_diff, y2y_diff, do_cumsum, rolling, cropland_only, var, das)