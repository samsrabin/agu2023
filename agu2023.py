# %%
import numpy as np
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import datetime as dt

# Import general CTSM Python utilities
import sys
my_ctsm_python_gallery = "/Users/sam/Documents/git_repos/ctsm_python_gallery_myfork/ctsm_py/"
sys.path.append(my_ctsm_python_gallery)
import utils

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

# Calculate total value (instead of per-area)
def get_total_value(dse, da, cropland_only):
    dse["tmp"] = da
    da = utils.grid_one_variable(dse, "tmp")
    old_units = da.attrs["units"]
    if "/m^2" in old_units:
        new_units = old_units.replace("/m^2", "")
    else:
        new_units = old_units + " m^2"
    area_da = dse["area"]*1e6
    if cropland_only:
        area_da = area_da * dse["frac_crop"]
    da *= area_da
    da.attrs["units"] = new_units
    
    return da

# Convert units
def convert_units(dse, da):
    if "gC" in da.attrs["units"]:
        units = da.attrs["units"].replace("gC", "PgC")
        da = da * 1e-15
        da.attrs["units"] = units
        
    if "/s" in da.attrs["units"]:
        t0 = dse["time_bounds"].values[1,0]
        t1 = dse["time_bounds"].values[1,1]
        tdelta = t1 - t0
        if tdelta  == dt.timedelta(days=365):
            units = da.attrs["units"].replace("/s", "/yr")
            da = da * 365*24*60*60
        else:
            raise RuntimeError(f"Unrecognized time delta: {tdelta}")
        da.attrs["units"] = units
    
    return da


def get_y2y_chg(v, da):
    attrs = da.attrs
    da1 = da.isel(time=slice(1,None))
    da0 = da.isel(time=slice(None,-1))
    da0 = da0.assign_coords({"time": da1["time"]})
    da = da1 - da0
    da.attrs = attrs
    da.name = f"{v} y2y flux"
    return da

def get_wtg_inds(cropland_only, var):
    if var == "GRAINC_TO_FOOD_ANN":
        if cropland_only:
            raise RuntimeError("GRAINC_TO_FOOD_ANN can't be used with cropland_only=True")
        wtg = "pfts1d_wtgcell"
        inds = "pfts1d_gi"
    elif "CROPPROD" in var:
        if cropland_only:
            raise RuntimeError(f"{var} can't be used with cropland_only=True")
        wtg = None
        inds = None
    else:
        if cropland_only:
            wtg = "cols1d_wtlunit"
            inds = "cols1d_gi"
        else:
            wtg = "cols1d_wtgcell"
            inds = "cols1d_gi"
    return wtg,inds


# Get fraction of each gridcell that's cropland
def get_frac_crop(dse):
    da = dse["land1d_wtgcell"].where(dse["land1d_ityplunit"] == 2)
    da = da.groupby(dse["land1d_gi"]).sum(skipna=True)
    da = da.rename({"land1d_gi": "gridcell"})
    dse["frac_crop_vector"] = da
    dse["frac_crop"] = utils.grid_one_variable(dse, "frac_crop_vector")
    
    return dse


def get_weighted(ds, cropland_only, v, var, wtg, inds, e):
    if cropland_only and v==0:
        # Get crop PFTs/columns
        ds[e] = get_only_cropland(ds[e], drop="time" not in ds[e][inds].dims)

        # Get fraction of each gridcell that's cropland
        ds[e] = get_frac_crop(ds[e])
            
    # Ensure that weights sum to 1
    groupby_var_da = ds[e][inds]
    for t in np.arange(ds[e].dims["time"]):
        dset = ds[e].isel(time=t)
        groupby_var_da = dset[inds]
        wts_grouped = dset[wtg].groupby(groupby_var_da)
        wtsum = wts_grouped.sum(skipna=True)
        if np.abs(np.nanmax(wtsum - 1)) > 1e-9:
            raise RuntimeError(f"Weights don't add to 1; range {np.nanmin(wtsum)} to {np.nanmax(wtsum)}")
        break # Just check the first timestep

    # Calculate weighted mean
    tmp = np.full((ds[e].dims["time"], ds[e].dims["gridcell"]), np.nan)
    for t in np.arange(ds[e].dims["time"]):
        dset = ds[e].isel(time=t, drop=True)
        dat_grouped = (dset[var] * dset[wtg]).groupby(dset[inds])
        dat = dat_grouped.sum(skipna=True)
        tmp[t,:] = dat.values
    new_coords = {"time": ds[e]["time"],
                  "gridcell": ds[e]["gridcell"]
                 }
    da = xr.DataArray(
        data=tmp,
        coords=new_coords,
        dims=new_coords,
        attrs=ds[e][var].attrs
    )
    
    return da


def make_plot(expt_list, ds, var_list, abs_diff, rel_diff, y2y_diff, cropland_only):
    
    if isinstance(var_list, str):
        var_list = [var_list]
    
    if rel_diff and y2y_diff:
        raise RuntimeError("rel_diff and y2y_diff are mutually exclusive")
    if rel_diff and abs_diff:
        raise RuntimeError("rel_diff and abs_diff are mutually exclusive")

    for v, var in enumerate(var_list):
        wtg, inds = get_wtg_inds(cropland_only, var)
                
        das = []
        plt.figure()
        for e, expt_name in enumerate(expt_list):
            
            if wtg is not None:
                da = get_weighted(ds, cropland_only, v, var, wtg, inds, e)
            else:
                ds[e] = get_frac_crop(ds[e])
                da = ds[e][var]
            
            # Calculate total value (instead of per-area)
            da = get_total_value(ds[e], da, cropland_only)
            
            # Calculate global sum
            da = da.sum(dim=["lat","lon"], keep_attrs=True)
            
            # Convert units
            da = convert_units(ds[e], da)
            
            # Get year-to-year change (i.e., net flux)
            if y2y_diff:
                da = get_y2y_chg(v, da)
            
            # Plot (or save for plotting later)
            units = da.attrs["units"]
            if rel_diff or abs_diff:
                das.append(da)
            else:
                da.plot()
        
        if rel_diff or abs_diff:
            for e in np.arange(1, len(das)):
                if rel_diff:
                    da = das[e] / das[0]
                elif abs_diff:
                    da = das[e] - das[0]
                da.plot()
            plt.legend(expt_list[1:])
        else:
            plt.legend(expt_list)
        if cropland_only:
            plt.title(var + " (cropland only)")
        else:
            plt.title(var)
        if rel_diff:
            plt.axhline(y=1, color="k", linestyle="--")
            plt.ylabel(f"Relative to {expt_list[0]}")
        elif y2y_diff or abs_diff:
            plt.axhline(y=0, color="k", linestyle="--")
            if abs_diff:
                plt.ylabel(f"Relative to {expt_list[0]} ({units})")
            else:
                plt.ylabel(units)
        else:
            plt.ylabel(units)
        plt.show()


# %% Import

os.chdir("/Users/Shared/CESM_runs/agu2023_10x15v3")

expt_list = ["Toff_Roff", "Thi_Rhi", "Toff_Roff_fromHi", "Thi_Rhi_fromOff", "Thi_Roff_fromOff", "Toff_Rhi_fromHi"]
# expt_list = ["Toff_Roff", "Thi_Rhi_fromOff"]
# expt_list = ["Thi_Rhi", "Toff_Roff_fromHi"]
# expt_list = ["Toff_Roff", "Thi_Rhi_fromOff", "Toff_Rhi_fromOff"]
# expt_list = ["Thi_Rhi", "Toff_Roff_fromHi", "Thi_Roff_fromHi"]
ds1 = []
ds2 = []
ds3 = []
for e, expt_name in enumerate(expt_list):
    file = glob.glob(f"*{expt_name}.clm2.h1s.*")
    if len(file) != 1:
        raise RuntimeError(f"{len(file)} matches found")
    ds1.append(xr.open_dataset(file[0]))
    file = glob.glob(f"*{expt_name}.clm2.h2s.*")
    if len(file) != 1:
        raise RuntimeError(f"{len(file)} matches found")
    ds2.append(xr.open_dataset(file[0]))
    file = glob.glob(f"*{expt_name}.clm2.h3s.*")
    if len(file) != 1:
        raise RuntimeError(f"{len(file)} matches found")
    ds3.append(xr.open_dataset(file[0]))


# %% h2 files
abs_diff = False
rel_diff = False
y2y_diff = False
cropland_only = True
# var_list = ["TOTLITC", "TOTLITC_1m", "TOTLITN", "TOTLITN_1m", "TOTSOMC", "TOTSOMC_1m", "TOTSOMN", "TOTSOMN_1m"]
var_list = ["TOTSOMC"]

make_plot(expt_list, ds2, var_list, abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=cropland_only)


# %% h3 files
abs_diff = False
rel_diff = False
y2y_diff = False
var_list = ["CROPPROD1C", "CROPPROD1C_LOSS"]

make_plot(expt_list, ds3, var_list, abs_diff=abs_diff, rel_diff=rel_diff, y2y_diff=y2y_diff, cropland_only=False)


# %% GRAINC_TO_FOOD_ANN
rel_diff = False

make_plot(expt_list, ds1, "GRAINC_TO_FOOD_ANN", rel_diff=rel_diff, cropland_only=False)



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