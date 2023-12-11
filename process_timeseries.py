# %%
import xarray as xr
import os
import pickle

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
expt_list = config.getlist("runs", "expt_list")


# %% Define things


def get_ts_dict(var_list, ds):
    ts_dict = {}
    for var in var_list:
        for cropland_only in [False, True]:
            # Get name of output timeseries DataArray
            if cropland_only:
                var_ts = f"{var}_croponly"
            else:
                var_ts = var

            try:
                wtg, inds = agu23.get_wtg_inds(cropland_only, var, ds)
            except Exception as inst:
                if str(inst) == f"{var} can't be used with cropland_only=True":
                    ts_dict[var_ts] = None
                    continue
                else:
                    raise

            print(f"    {var}, cropland_only=={cropland_only}")
            da = agu23.get_timeseries_da(ds, cropland_only, var, wtg, inds)

            da.name = var_ts
            ts_dict[var_ts] = da

    return ts_dict


var_list_list = [
    ["SOILC_HR", "NEP", "NEE", "NBP"],
    ["GRAINC_TO_FOOD_ANN"],
    [
        "TOTLITC",
        "TOTLITC_1m",
        "TOTLITN",
        "TOTLITN_1m",
        "TOTSOMC",
        "TOTSOMC_1m",
        "TOTSOMN",
        "TOTSOMN_1m",
    ],
    ["CROPPROD1C", "CROPPROD1C_LOSS"],
]


# %% Process timeseries for all experiments

for e, expt_name in enumerate(expt_list):
    print(f"{expt_name}: ")

    for v, var_list in enumerate(var_list_list):
        # Read file
        file_in = agu23.get_file(v, expt_name)
        ds = xr.open_dataset(file_in)

        # Get dictionary with timeseries for each variable (croponly true and false, if possible)
        ts_dict = get_ts_dict(var_list, ds)

        # Save as pickle file
        file_out = file_in + ".pickle"
        with open(file_out, "wb") as handle:
            pickle.dump(ts_dict, handle, protocol=4)

        ds.close()
    break
