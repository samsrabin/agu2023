import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
import datetime as dt
import configparser
import os
import sys
import cftime
import mapsmod as mm


def read_config():
    scriptsDir = os.path.dirname(__file__)
    config = configparser.ConfigParser(
        converters={"list": lambda x: [i.strip() for i in x.split(",")]}
    )
    config.read(os.path.join(scriptsDir, "agu2023.ini"))
    return config


# Read configuration
config = read_config()

# Import general CTSM Python utilities
sys.path.append(config["system"]["my_ctsm_python_gallery"])
import utils


def get_file(h, expt_name):
    pattern = f"*{expt_name}.clm2.h{h}s.*.nc"
    file = glob.glob(pattern)
    if len(file) > 1:
        raise RuntimeError(f"{len(file)} matches found: {file}")
    elif len(file) < 1:
        raise RuntimeError(f"{len(file)} matches found for {pattern}")
    return file[0]


def get_only_cropland(dse, drop):
    # Drop pfts1d variables, which aren't needed. This speeds things up.
    drop_list = []
    for v in dse:
        if "pfts1d_" in v:
            drop_list.append(v)
    dse = dse.drop(drop_list)

    # Get dict of original dimensions for each variable
    orig_var_dims = {}
    for v in dse:
        orig_var_dims[v] = dse[v].copy().dims

    # Include only crop landunits
    for v in dse:
        if "column" in dse[v].dims:
            dse[v] = dse[v].where(dse["cols1d_itype_lunit"] == 2, drop=drop)

    # Remove extraneous dimension(s) added to various variables
    for v in dse:
        for d in dse[v].dims:
            if d not in orig_var_dims[v]:
                dse[v] = dse[v].assign_coords({d: dse[d]})
                dse[v] = dse[v].isel({d: 0}, drop=True)
    return dse


def get_only_cropland2(dse, wtg):
    where_not_crop = np.where(dse["cols1d_itype_lunit"].values != 2)
    wtg_vals = dse[wtg].values
    wtg_vals[where_not_crop] = 0
    wtg_da = xr.DataArray(
        data=wtg_vals,
        coords=dse[wtg].coords,
        dims=dse[wtg].dims,
        attrs=dse[wtg].attrs,
    )
    dse[wtg] = wtg_da
    return dse


def grid_da(ds, da):
    ds["tmp"] = da
    da = utils.grid_one_variable(ds, "tmp")
    return da


# Calculate total value (instead of per-area)
def get_total_value(dse, da_in, cropland_only):
    old_units = da_in.attrs["units"]
    if "/m^2" in old_units:
        new_units = old_units.replace("/m^2", "")
    else:
        new_units = old_units + " m^2"
    area_da = dse["area"] * dse["landfrac"] * 1e6
    if cropland_only:
        area_da = area_da * dse["frac_crop"]
    da = da_in * area_da
    da.attrs["units"] = new_units

    return da


# Convert units
def convert_units(dse, da):
    units = da.attrs["units"]

    if "gC/m^2" in units or "gN/m^2" in units:
        units = units.replace("gC/m^2", "tC/ha")
        units = units.replace("gN/m^2", "tC/ha")
        da = da * 1e-6 * 1e4

    if "gC" in units or "gN" in units:
        units = units.replace("gC", "PgC")
        units = units.replace("gN", "PgN")
        da = da * 1e-15

    if "/s" in units:
        t0 = dse["time_bounds"].values[1, 0]
        t1 = dse["time_bounds"].values[1, 1]
        tdelta = t1 - t0
        if tdelta == dt.timedelta(days=365):
            units = units.replace("/s", "/yr")
            da = da * 365 * 24 * 60 * 60
        else:
            raise RuntimeError(f"Unrecognized time delta: {tdelta}")

    da.attrs["units"] = units

    return da


def get_y2y_chg(da):
    attrs = da.attrs
    da1 = da.isel(time=slice(1, None))
    da0 = da.isel(time=slice(None, -1))
    da0 = da0.assign_coords({"time": da1["time"]})
    da = da1 - da0
    da.attrs = attrs
    return da


def get_wtg_inds(cropland_only, var, ds):
    if cropland_only and (var == "GRAINC_TO_FOOD_ANN" or "gridcell" in ds[var].dims):
        raise RuntimeError(f"{var} can't be used with cropland_only=True")

    dims = ds[var].dims
    if "pft" in dims:
        wtg = "pfts1d_wtgcell"
        inds = "pfts1d_gi"
    elif "gridcell" in dims:
        wtg = None
        inds = None
    elif "column" in dims:
        if cropland_only:
            wtg = "cols1d_wtlunit"
            inds = "cols1d_gi"
        else:
            wtg = "cols1d_wtgcell"
            inds = "cols1d_gi"
    else:
        raise RuntimeError(f"Unknown wtg/inds for dims {dims}")
    return wtg, inds


# Get fraction of each gridcell that's cropland
def get_frac_crop(dse):
    da = dse["land1d_wtgcell"].where(dse["land1d_ityplunit"] == 2)
    da = da.groupby(dse["land1d_gi"]).sum(skipna=True)
    da = da.rename({"land1d_gi": "gridcell"})
    dse["frac_crop_vector"] = da
    dse["frac_crop"] = utils.grid_one_variable(dse, "frac_crop_vector")

    return dse


def get_weighted(dse, cropland_only, var, wtg, inds):
    # Get crop PFTs/columns
    if cropland_only and any(["pfts1d_" in v for v in dse]):
        dse = get_only_cropland(dse, drop="time" not in dse[inds].dims)

    # Get fraction of each gridcell that's cropland
    if cropland_only and "frac_crop" not in dse:
        dse = get_frac_crop(dse)

    # Ensure that weights sum to 1
    groupby_var_da = dse[inds]
    for t in np.arange(dse.dims["time"]):
        dset = dse.isel(time=t)
        groupby_var_da = dset[inds]
        wts_grouped = dset[wtg].groupby(groupby_var_da)
        wtsum = wts_grouped.sum(skipna=True)
        if np.abs(np.nanmax(wtsum - 1)) > 1e-9:
            raise RuntimeError(
                f"Weights don't add to 1; range {np.nanmin(wtsum)} to {np.nanmax(wtsum)}"
            )
        break  # Just check the first timestep

    # Calculate weighted mean
    tmp = np.full((dse.dims["time"], dse.dims["gridcell"]), np.nan)
    for t in np.arange(dse.dims["time"]):
        dset = dse.isel(time=t, drop=True)
        dat_grouped = (dset[var] * dset[wtg]).groupby(dset[inds])
        dat = dat_grouped.sum(skipna=True)  # BOTTLENECK
        tmp[t, :] = dat.values
    new_coords = {"time": dse["time"], "gridcell": dse["gridcell"]}
    da = xr.DataArray(
        data=tmp, coords=new_coords, dims=new_coords, attrs=dse[var].attrs
    )

    return dse, da


def process_and_make_plot(
    expt_list, ds, var_list, abs_diff, rel_diff, y2y_diff, cropland_only, rolling=None
):
    if isinstance(var_list, str):
        var_list = [var_list]

    if rel_diff and y2y_diff:
        raise RuntimeError("rel_diff and y2y_diff are mutually exclusive")
    if rel_diff and abs_diff:
        raise RuntimeError("rel_diff and abs_diff are mutually exclusive")

    for _, var in enumerate(var_list):
        # Process modifiers
        do_cumsum = False
        while "." in var:
            if ".CUMSUM" in var:
                do_cumsum = True
                var = var.replace(".CUMSUM", "")
            else:
                raise RuntimeError(f"Unexpected modifier(s) in var: {var}")

        wtg, inds = get_wtg_inds(cropland_only, var, ds[0])

        das = []
        for dse in ds:
            da = get_timeseries_da(dse, cropland_only, var, wtg, inds)
            da = shift_1_year_earlier(da)
            das.append(da)

        make_ts_plot(
            expt_list,
            abs_diff,
            rel_diff,
            y2y_diff,
            do_cumsum,
            rolling,
            cropland_only,
            var,
            das,
        )


def make_ts_plot(
    expt_list,
    abs_diff,
    rel_diff,
    y2y_diff,
    do_cumsum,
    rolling,
    cropland_only,
    var,
    das_in,
    title=None,
    figsize=None,
    axlabelsize=None,
    titlesize=None,
    ticklabelsize=None,
    legendsize=None,
    xticks=None,
    xticklabels=None,
    colors=None,
    styles=None,
    file_out=None,
    show=False,
):
    # Ensure all DataArrays have the same units
    check_consistent_units(das_in)

    # Modify DataArrays, if needed
    das = []
    for d, da in enumerate(das_in):
        expt = expt_list[d]
        if "from" in expt:
            hist_expt_name = get_hist_expt_name(expt)
            da0 = das_in[expt_list.index(hist_expt_name)]
            da2 = xr.concat(
                (da0.sel(time=slice("1901-01-01", "2014-12-31")), da), dim="time"
            )
            da3 = modify_timeseries_da(da2, do_cumsum, rolling, y2y_diff)
            das.append(da3.sel(time=slice("2015-01-01", "2100-12-31")))
        else:
            das.append(modify_timeseries_da(da, do_cumsum, rolling, y2y_diff))
    units = das[0].attrs["units"]

    # Get styles to cycle through
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if styles is None:
        styles = "-" * len(expt_list)
    linewidth = 2

    if rel_diff or abs_diff:
        for e in np.arange(1, len(das)):
            if rel_diff:
                da = das[e] / das[0]
            elif abs_diff:
                da = das[e] - das[0]
            das[e] = da
        if rel_diff:
            das[0] = das[0] / das[0]
        else:
            das[0] = das[0] - das[0]

    plt.figure(figsize=figsize)

    axline_color = "#" + "d" * 6
    axline_style = "-"
    if rel_diff:
        plt.axhline(y=1, color=axline_color, linestyle=axline_style, label="_nolegend_")
    elif y2y_diff or abs_diff:
        plt.axhline(y=0, color=axline_color, linestyle=axline_style, label="_nolegend_")

    for e, expt_name in enumerate(expt_list):
        da = das[e].copy()
        if "from" in expt_name:
            hist_expt_name = get_hist_expt_name(expt_name)
            da_hist = das[expt_list.index(hist_expt_name)].copy()
            da_hist = da_hist.sel(time=slice("2014-01-01", "2014-12-31"))
            da = xr.concat((da_hist, da), dim="time")
        da.plot(color=colors[e], linestyle=styles[e], linewidth=linewidth)
    legend_items = expt_list

    plt.legend(legend_items, fontsize=legendsize)

    # Set title
    if title is None:
        title = var
    if cropland_only:
        title += " (cropland only)"
    if do_cumsum:
        title = "Cumulative " + title
        units = units.replace(" (cumulative)", "")
    plt.title(title, fontsize=titlesize)

    if rel_diff:
        plt.ylabel(f"Relative to {expt_list[0]} (unitless)", fontsize=axlabelsize)
    elif y2y_diff or abs_diff:
        if abs_diff:
            plt.ylabel(f"Relative to {expt_list[0]} ({units})", fontsize=axlabelsize)
        else:
            plt.ylabel(units, fontsize=axlabelsize)
    else:
        plt.ylabel(units, fontsize=axlabelsize)
    plt.xlabel("Year", fontsize=axlabelsize)
    plt.tick_params(labelsize=ticklabelsize)

    if xticks is not None:
        plt.xticks(xticks, xticklabels)

    # Save, if doing so
    if file_out is not None:
        if os.path.exists(file_out) and not confirm(f"{file_out} exists. Overwrite?"):
            print("Skipping figure save.")
        else:
            if not os.path.exists(file_out):
                print(f"Saving {file_out}")
            plt.savefig(file_out)

    if show:
        plt.show()


def check_consistent_units(das_in):
    units = das_in[0].attrs["units"]
    for d, da in enumerate(das_in):
        if d == 0:
            continue
        if units != da.attrs["units"]:
            raise RuntimeError(f"Units mismatch: {units} vs. {da.attrs['units']}")


def make_maps_plot(
    expt_list,
    abs_diff,
    rel_diff,
    cropland_only,
    var,
    das,
    title=None,
    figsize=None,
    axlabelsize=None,
    titlesize=None,
    ticklabelsize=None,
    legendsize=None,
    file_out=None,
    show=False,
):
    # Ensure all DataArrays have the same units
    check_consistent_units(das)
    units = das[0].attrs["units"]

    if rel_diff or abs_diff:
        da_absmax = None
        for e in np.arange(1, len(expt_list)):
            da = das[e].copy()
            da = da - das[0]
            if rel_diff:
                da /= das[0]
            elif not abs_diff:
                raise RuntimeError("???")
            if e==1:
                da_absmax = np.nanmax(np.abs(da.values))
            else:
                da_absmax = np.nanmax((da_absmax, np.nanmax(np.abs(da.values))))
            das[e] = da
        if rel_diff:
            vrange = mm.limit_map_outliers(da_absmax)
    else:
        vrange = None

    Nexpt = len(expt_list)
    if Nexpt == 4:
        ny = 2
        nx = 2
    else:
        raise RuntimeError(f"Specify subplot layout for Nexpt = {Nexpt}")
    fig = plt.figure(figsize=figsize)

    ims = []
    for e, expt in enumerate(expt_list):
        ax = mm.make_axis(fig, ny, nx, e + 1)
        if e==0 or not (rel_diff or abs_diff):
            this_units = units
            this_vrange = None
            cmap = mm.cropcal_colors["seq_other"]
        else:
            if abs_diff:
                this_units = f"rel. to {expt_list[0]} ({units})"
                vrange = None
            elif rel_diff:
                this_units = f"rel. to {expt_list[0]} (unitless)"
                this_vrange = vrange
            cmap = mm.cropcal_colors["div_yieldirr"]
        if e==0 and (rel_diff or abs_diff) and np.nanmin(das[e]) < 0:
            cmap = "RdBu"
        im, _ = mm.make_map(ax, das[e], cmap=cmap, show_cbar=True, this_title=expt, units=this_units, vrange=this_vrange)
        ims.append(im)

    if abs_diff:
        mm.equalize_colorbars(ims[1:], center0=True)
    elif not rel_diff:  # Already equalized in limit_map_outliers()
        mm.equalize_colorbars(ims)

    # Save, if doing so
    if file_out is not None:
        if os.path.exists(file_out) and not confirm(f"{file_out} exists. Overwrite?"):
            print("Skipping figure save.")
        else:
            if not os.path.exists(file_out):
                print(f"Saving {file_out}")
            plt.savefig(file_out, dpi=300, bbox_inches="tight")

    if show:
        plt.show()


def get_maps_da(dse, cropland_only, var, wtg, inds):
    if wtg is not None:
        dse, da_perarea = get_weighted(dse, cropland_only, var, wtg, inds)
    else:
        dse = get_frac_crop(dse)
        da_perarea = dse[var]

    # Grid
    da_perarea = grid_da(dse, da_perarea)

    # Calculate total value (instead of per-area)
    units = da_perarea.attrs["units"]
    da = get_total_value(dse, da_perarea, cropland_only)
    if units != da_perarea.attrs["units"]:
        raise RuntimeError("???")

    # Convert units
    da_perarea = convert_units(dse, da_perarea)
    da = convert_units(dse, da)

    # Ignore first time step, which seems to be garbage for NBP etc.
    Ntime = dse.dims["time"]
    da_perarea = da_perarea.isel(time=slice(1, Ntime))
    da = da.isel(time=slice(1, Ntime))

    return da_perarea, da


def get_timeseries_da(dse, cropland_only, var, wtg, inds):
    _, da = get_maps_da(dse, cropland_only, var, wtg, inds)

    return da.sum(dim=["lat", "lon"], keep_attrs=True)


def modify_timeseries_da(da, do_cumsum, rolling, y2y_diff):
    if do_cumsum:
        units = da.attrs["units"]
        if "/yr" not in units:
            raise RuntimeError(f"Trying to do cumsum of units without /yr: {units}")
        da = da.cumsum(dim="time", keep_attrs=True)
        da.attrs["units"] = units.replace("/yr", " (cumulative)")

    # Smooth
    if rolling is not None:
        da = da.rolling(time=rolling, center=True).mean()

    # Get year-to-year change (i.e., net flux)
    if y2y_diff:
        da = get_y2y_chg(da)

    return da


def shift_1_year_earlier(da):
    years = [t.year for t in da["time"].values]
    new_time = [
        cftime.DatetimeNoLeap(y - 1, 1, 1, 0, 0, 0, 0, has_year_zero=True)
        for y in years
    ]
    new_time_da = xr.DataArray(
        data=new_time,
        attrs=da["time"].attrs,
        dims=["time"],
    )
    da = da.assign_coords(time=new_time_da)
    return da


def get_hist_expt_name(expt_name):
    if "fromOff" in expt_name:
        hist_expt_name = "Toff_Roff"
    elif "fromHiLo" in expt_name:
        hist_expt_name = "Thi_Rlo"
    elif "fromHi" in expt_name:
        hist_expt_name = "Thi_Rhi"
    else:
        raise RuntimeError(f'Unrecognized "from" in expt_name: {expt_name}')

    return hist_expt_name


def confirm(message):
    while True:
      answer = input(message + " (y/N) ")
      answer = answer.lower()
      if answer == "y":
        return True
      elif answer == "n" or answer == "":
        return False
      print("Please answer with 'y' for yes or 'n' for no.")

