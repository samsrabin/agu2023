# %%
import xarray as xr
import os
import pickle
import configparser
import glob

# Import supporting module
import sys

scriptsDir = os.path.dirname(__file__)
sys.path.append(scriptsDir)
import agu2023mod as agu23


# %% Define things


def get_maps_dict(var_list, ds):
    maps_dict = {}
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
                    continue
                else:
                    raise

            print(f"    {var}, cropland_only=={cropland_only}")
            da_perarea, da = agu23.get_maps_da(ds, cropland_only, var, wtg, inds)

            da_perarea.name = var_ts + "_perarea"
            maps_dict[da_perarea.name] = da_perarea

            da.name = var_ts
            maps_dict[da.name] = da

    return maps_dict


def read_process_maps_config():
    scriptsDir = os.path.dirname(__file__)
    scriptsDir = os.path.abspath(scriptsDir)
    o = configparser.ConfigParser(
        converters={"list": lambda x: [i.strip() for i in x.split(",")]},
    )
    o.read(os.path.join(scriptsDir, "process_maps.ini"))
    os.chdir(o["DEFAULT"]["this_dir"])
    expt_list = o.getlist("DEFAULT", "expt_list")

    file_list = glob.glob(os.path.join(scriptsDir, "process_maps_h*.ini"))
    file_list.sort()
    # Loop through h files; requires that no h# be skipped
    var_list_list = []
    for file in file_list:
        var_list_list.append
        o.read(file)
        var_list_list.append(o.getlist("DEFAULT", "var_list"))
    return expt_list, var_list_list


# %% Process maps for all experiments

expt_list, var_list_list = read_process_maps_config()

for e, expt_name in enumerate(expt_list):
    print(f"{expt_name}: ")

    maps_dict = {}
    for v, var_list in enumerate(var_list_list):
        # Read file
        file_in = agu23.get_file(v, expt_name)
        ds = xr.open_dataset(file_in)

        # Get dictionary with maps for each variable (croponly true and false, if possible)
        maps_dict.update(get_maps_dict(var_list, ds))

    # Save as pickle file
    file_out = file_in + ".maps.pickle"
    file_out = file_out.replace(f".h{v}s", "")
    with open(file_out, "wb") as handle:
        pickle.dump(maps_dict, handle, protocol=4)

    ds.close()
