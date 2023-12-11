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

            # Ignore extra years
            if da is None:
                da = da_tmp.sel(time=slice(f"{y1}-01-01", f"{yN}-12-31"))
                if len(da["time"].values) == 0:
                    da = None
                    continue
            else:
                # Assuming files are in increasing chronological order
                yN_prev = da["time"].values[-1].year
                da_tmp = da_tmp.sel(time=slice(f"{yN_prev}-01-01", f"{yN}-12-31"))
                da = xr.concat((da, da_tmp), dim="time")

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
    expt_list = o.getlist("DEFAULT", "expt_list")
    fig_keys = [k for k in o["fig"].keys()]
    if "rolling" not in fig_keys:
        rolling = None
    else:
        rolling = o.getint("fig", "rolling")
    var_keys = [k for k in o["DEFAULT"].keys()]
    if "title" not in var_keys or o["DEFAULT"]["title"] == "":
        title = None
    else:
        title = o["DEFAULT"]["title"]

    xticks, xticklabels = get_xticks([int(x) for x in o.getlist("fig", "xticks")])

    colors, styles = get_styles(expt_list)

    figsize = (
        o.getfloat("fig", "figsize_x"),
        o.getfloat("fig", "figsize_y"),
    )

    if "fig_dir" not in o["DEFAULT"].keys():
        o["DEFAULT"]["fig_dir"] = o["DEFAULT"]["this_dir"]
    basename_noext, _ = os.path.splitext(os.path.basename(ini_file))
    file_out = os.path.join(o["DEFAULT"]["fig_dir"], basename_noext + ".pdf")

    return (
        o,
        expt_list,
        rolling,
        title,
        xticks,
        xticklabels,
        colors,
        styles,
        figsize,
        file_out,
    )


def get_colors(expt_list, colors):
    new_colors = []
    for expt_name in expt_list:
        is_original_clm = re.match(".*Toff_Roff", expt_name)
        if is_original_clm:
            new_colors.append("#000000")
        else:
            new_colors.append("#" + colors.pop(0))
    return new_colors


def get_styles(expt_list):
    colors = []
    styles = []
    for expt in expt_list:
        # Get color
        if "Toff" in expt:
            colors.append("black")
        elif "Thi" in expt:
            colors.append("peru")
        else:
            raise RuntimeError(f"Unable to parse tillage setting from '{expt}'")

        # Get style
        if "Roff" in expt:
            styles.append("-")
        elif "Rlo" in expt:
            styles.append("--")
        elif "Rhi" in expt:
            styles.append((0, (5, 5)))
        else:
            raise RuntimeError(f"Unable to parse tillage setting from '{expt}'")
    return colors, styles


# %% Make plot


def main(ini_file):
    (
        o,
        expt_list,
        rolling,
        title,
        xticks,
        xticklabels,
        colors,
        styles,
        figsize,
        file_out,
    ) = read_fig_config(ini_file)
    this_dir = o["DEFAULT"]["this_dir"]
    os.chdir(this_dir)

    das = get_das(
        expt_list,
        o["DEFAULT"]["name"],
        o.getboolean("fig", "cropland_only"),
        o.getint("fig", "y1"),
        o.getint("fig", "yN"),
    )

    agu23.make_ts_plot(
        expt_list,
        o.getboolean("fig", "abs_diff"),
        o.getboolean("fig", "rel_diff"),
        o.getboolean("fig", "y2y_diff"),
        o.getboolean("fig", "do_cumsum"),
        rolling,
        o.getboolean("fig", "cropland_only"),
        o["DEFAULT"]["name"],
        das,
        title=title,
        figsize=figsize,
        axlabelsize=o["fig"]["axlabelsize"],
        titlesize=o["fig"]["titlesize"],
        ticklabelsize=o["fig"]["ticklabelsize"],
        legendsize=o["fig"]["legendsize"],
        xticks=xticks,
        xticklabels=xticklabels,
        colors=colors,
        styles=styles,
        file_out=file_out,
        show=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ini_file", type=str, nargs=1)
    args = parser.parse_args()
    main(args.ini_file[0])

# main("ini/test.ini")
