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
expt_list = config.getlist("runs", "expt_list")


# %% Pack pickles

for expt_name in expt_list:

    # Get list of files
    pattern_yearrange = "[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]"
    pattern = f"*{expt_name}.clm2.h[0-9]s.{pattern_yearrange}.nc.pickle"
    file_list = glob.glob(pattern)
    file_list.sort()

    # Combine
    dict_out = {}
    for f in file_list:
        with open(f, "rb") as handle:
            dict_in = pickle.load(handle)
        dict_out.update(dict_in)

    # Save
    expt_name = f.split(".")[0]
    yearrange = f.split(".")[3]
    file_out = f"{expt_name}.clm2.{yearrange}.pickle"
    with open(file_out, "wb") as handle:
        pickle.dump(dict_out, handle, protocol=4)
