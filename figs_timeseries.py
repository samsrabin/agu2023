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


def get_das(expt_list, var, cropland_only, y1, yN):
    if cropland_only:
        var += "_croponly"
    pattern_yearrange = "[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]"
    das = []
    for expt in expt_list:
        pattern = f"*{expt}.clm2.{pattern_yearrange}.nc.maps.pickle"
        file_in = glob.glob(pattern)
        if len(file_in) > 1:
            raise RuntimeError(f"Found {len(file_in)} matches: {file_in}")
        elif len(file_in) == 0:
            raise RuntimeError(f"Found 0 matches for {pattern}")

        with open(file_in[0], "rb") as handle:
            dict_in = pickle.load(handle)
        da = dict_in[var]

        # Shift 1 year earlier
        da = agu23.shift_1_year_earlier(da)

        # Ignore extra years
        da = da.sel(time=slice(f"{y1}-01-01", f"{yN}-12-31"))

        # Get global sum
        da = da.sum(dim=["lon", "lat"], keep_attrs=True)

        das.append(da)

    return das


def get_xticks(yearlist):
    xticks = []
    xticklabels = []
    for y in yearlist:
        xticks.append(cftime.DatetimeNoLeap(y, 1, 1, 0, 0, 0, 0, has_year_zero=True))
        xticklabels.append(y)
    return xticks, xticklabels


def read_fig_config(ini_file):
    scriptsDir = os.path.dirname(__file__)
    o = configparser.ConfigParser(
        converters={"list": lambda x: [i.strip() for i in x.split(",")]},
        allow_no_value=True,
    )
    o.read(os.path.join(scriptsDir, ini_file))
    expt_list = o.getlist("runs", "expt_list")
    fig_keys = [k for k in o["fig"].keys()]
    if "rolling" not in fig_keys:
        rolling = None
    else:
        rolling = o.getint("fig", "rolling")
    var_keys = [k for k in o["var"].keys()]
    if "title" not in var_keys or o["var"]["title"] == "":
        title = None
    else:
        title = o["var"]["title"]

    xticks, xticklabels = get_xticks([int(x) for x in o.getlist("fig", "xticks")])

    new_colors = get_colors(expt_list, o.getlist("fig", "colors"))

    figsize = (
        o.getfloat("fig", "figsize_x"),
        o.getfloat("fig", "figsize_y"),
    )

    return o, expt_list, rolling, title, xticks, xticklabels, new_colors, figsize


def get_colors(expt_list, colors):
    new_colors = []
    for expt_name in expt_list:
        is_original_clm = re.match(".*Toff_Roff", expt_name)
        if is_original_clm:
            new_colors.append("#000000")
        else:
            new_colors.append("#" + colors.pop(0))
    return new_colors


# %% Make plot


def main(ini_file):
    (
        o,
        expt_list,
        rolling,
        title,
        xticks,
        xticklabels,
        new_colors,
        figsize,
    ) = read_fig_config(ini_file)
    this_dir = o["runs"]["this_dir"]
    os.chdir(this_dir)

    das = get_das(
        expt_list,
        o["var"]["name"],
        o.getboolean("fig", "cropland_only"),
        o.getint("fig", "y1"),
        o.getint("fig", "yN"),
    )

    agu23.make_plot(
        expt_list,
        o.getboolean("fig", "abs_diff"),
        o.getboolean("fig", "rel_diff"),
        o.getboolean("fig", "y2y_diff"),
        o.getboolean("fig", "do_cumsum"),
        rolling,
        o.getboolean("fig", "cropland_only"),
        o["var"]["name"],
        das,
        title=title,
        figsize=figsize,
        axlabelsize=o["fig"]["axlabelsize"],
        titlesize=o["fig"]["titlesize"],
        ticklabelsize=o["fig"]["ticklabelsize"],
        legendsize=o["fig"]["legendsize"],
        xticks=xticks,
        xticklabels=xticklabels,
        colors=new_colors,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ini_file", type=str, nargs=1)
    args = parser.parse_args()
    main(args.ini_file[0])

# main("figs_timeseries.ini")
