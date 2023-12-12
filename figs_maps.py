# %%
import xarray as xr
import os
import pickle
import glob
import configparser
import cftime
import re
import argparse

# Import supporting module
import sys

scriptsDir = os.path.dirname(__file__)
sys.path.append(scriptsDir)
import agu2023mod as agu23


# %% Define


def get_slice(y1, yN):
    return slice(f"{y1}-01-01", f"{yN}-12-31")


def get_das(expt_list, var, do_cumsum, cropland_only, p1, pN):
    if cropland_only:
        var += "_croponly"
    pattern_yearrange = "[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]"
    das = []
    for e, expt in enumerate(expt_list):
        pattern = f"*{expt}_[12]*.clm2.{pattern_yearrange}.nc.maps.pickle"
        file_list = glob.glob(pattern)
        file_list.sort()
        if len(file_list) == 0:
            raise RuntimeError(f"Found 0 matches for {pattern}")

        da = None
        for file_in in file_list:
            with open(file_in, "rb") as handle:
                dict_in = pickle.load(handle)
            da_tmp = dict_in[var]

            # Shift 1 year earlier
            da_tmp = agu23.shift_1_year_earlier(da_tmp)

            # Concatenate
            if da is not None:
                # Assuming files are in increasing chronological order
                yN_prev = da["time"].values[-1].year
                da_tmp = da_tmp.sel(time=get_slice(yN_prev, 9999))
                da = xr.concat((da, da_tmp), dim="time")
            else:
                da = da_tmp

        # Process time dimension
        if do_cumsum:
            if pN is not None:
                raise RuntimeError("Code up do_cumsum with pN")
            da = da.copy().cumsum(dim="time", keep_attrs=True)
            if "time" in da.dims:
                raise RuntimeError("cumsum() kept time dimension???")
        else:
            da_p1 = da.sel(time=get_slice(p1[0], p1[1])).mean(dim="time", keep_attrs=True)
            if pN is not None:
                da_pN = da.sel(time=get_slice(pN[0], pN[1])).mean(
                    dim="time", keep_attrs=True
                )
                da = da_pN - da_p1
                da.attrs["units"] = da_p1.attrs["units"]
            else:
                da = da_p1

        # Convert units
        if da.attrs["units"] == "tC/ha":
            grain_cfrac = 0.45
            da /= grain_cfrac
            da.attrs["units"] = "tDM/ha"
            da = da.where(da > 0)

        das.append(da)

    return das


def get_xticks(yearlist):
    xticks = []
    xticklabels = []
    for y in yearlist:
        xticks.append(cftime.DatetimeNoLeap(y, 1, 1, 0, 0, 0, 0, has_year_zero=True))
        xticklabels.append(y)
    return xticks, xticklabels


def get_period_mean(period_str):
    return [int(y) for y in period_str.split("-")]


def read_fig_config(ini_file):
    scriptsDir = os.path.dirname(__file__)
    o = configparser.ConfigParser(
        converters={"list": lambda x: [i.strip() for i in x.split(",")]},
        allow_no_value=True,
    )
    ini_file_path = os.path.join(scriptsDir, ini_file)
    ini_file_path = os.path.realpath(ini_file_path)
    if not os.path.exists(ini_file_path):
        raise FileNotFoundError(f"Config file: '{ini_file_path}'")
    o.read(ini_file_path)
    expt_list = o.getlist("DEFAULT", "expt_list")
    fig_keys = [k for k in o["fig"].keys()]
    var_keys = [k for k in o["DEFAULT"].keys()]
    if "title" not in var_keys or o["DEFAULT"]["title"] == "":
        title = None
    else:
        title = o["DEFAULT"]["title"]

    figsize = (
        o.getfloat("fig", "figsize_x"),
        o.getfloat("fig", "figsize_y"),
    )

    if "fig_dir" not in o["DEFAULT"].keys():
        o["DEFAULT"]["fig_dir"] = o["DEFAULT"]["this_dir"]
    basename_noext, _ = os.path.splitext(os.path.basename(ini_file))
    file_out = os.path.join(o["DEFAULT"]["fig_dir"], basename_noext + ".png")

    p1 = get_period_mean(o["fig"]["p1"])
    if "pN" in fig_keys:
        pN = get_period_mean(o["fig"]["pN"])
    else:
        pN = None

    return o, expt_list, title, figsize, file_out, p1, pN


# %% Make plot


def main(ini_file):
    (
        o,
        expt_list,
        title,
        figsize,
        file_out,
        p1,
        pN,
    ) = read_fig_config(ini_file)
    this_dir = o["DEFAULT"]["this_dir"]
    os.chdir(this_dir)

    das = get_das(
        expt_list,
        o["DEFAULT"]["name"],
        o.getboolean("fig", "do_cumsum"),
        o.getboolean("fig", "cropland_only"),
        p1,
        pN,
    )

    agu23.make_maps_plot(
        expt_list,
        o.getboolean("fig", "abs_diff"),
        o.getboolean("fig", "rel_diff"),
        o.getboolean("fig", "cropland_only"),
        o["DEFAULT"]["name"],
        das,
        title=title,
        figsize=figsize,
        axlabelsize=o["fig"]["axlabelsize"],
        titlesize=o["fig"]["titlesize"],
        ticklabelsize=o["fig"]["ticklabelsize"],
        legendsize=o["fig"]["legendsize"],
        file_out=file_out,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ini_file", type=str, nargs=1)
    args = parser.parse_args()
    main(args.ini_file[0])

# main("ini/test.ini")
