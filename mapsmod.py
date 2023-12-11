import numpy as np
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.collections as mplcol
from matplotlib import cm
import matplotlib.colors as mcolors

# Ignore these annoying warnings
import warnings
warnings.filterwarnings("ignore", message="__len__ for multi-part geometries is deprecated and will be removed in Shapely 2.0. Check the length of the `geoms` property instead to get the  number of parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="Iteration over multi-part geometries is deprecated and will be removed in Shapely 2.0. Use the `geoms` property to access the constituent parts of a multi-part geometry.")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.")


# Colormaps (maps)
cropcal_colors = {
    "seq_timeofyear": "twilight_shifted",
    "seq_other": "plasma",  # magma_r? CMRmap_r?
    "div_yieldirr": "BrBG",
    "div_timeofyear": "twilight_shifted",
    "div_other_nonnorm": "PuOr_r",
    "div_other_norm": "RdBu_r",
    "underlay": [0.75, 0.75, 0.75, 1],
    "underlay_lighter": [0.85, 0.85, 0.85, 1],
    "underlay_lightest": [0.92, 0.92, 0.92, 1],
}

fontsize = {'axis_label': 28,
            'legend': 28,
            'suptitle': 36,
            'title': 30,
            'tick_label': 24}

fontsize['titles'] = 18
fontsize['axislabels'] = 14
fontsize['ticklabels'] = 14
fontsize['suptitle'] = 22


def make_map(
    ax,
    this_map,
    bounds=None,
    cbar=None,
    cbar_labelpad=4.0,
    cbar_max=None,
    cbar_spacing="uniform",
    cmap=cropcal_colors["seq_other"],
    extend_bounds="both",
    extend_nonbounds="both",
    linewidth=1.0,
    lonlat_bin_width=None,
    show_cbar=False,
    subplot_label=None,
    this_title=None,
    ticklabels=None,
    ticklocations=None,
    underlay=None,
    underlay_color=None,
    units=None,
    vmax=None,
    vmin=None,
    vrange=None,
):
    if underlay is not None:
        if underlay_color is None:
            underlay_color = cropcal_colors["underlay"]
        underlay_cmap = mcolors.ListedColormap(np.array([underlay_color, [1, 1, 1, 1]]))
        ax.pcolormesh(underlay.lon.values, underlay.lat.values, underlay, cmap=underlay_cmap)

    if bounds is not None:
        norm = mcolors.BoundaryNorm(bounds, cmap.N, extend=extend_bounds)
        im = ax.pcolormesh(
            this_map.lon.values, this_map.lat.values, this_map, shading="auto", norm=norm, cmap=cmap
        )
    else:
        im = ax.pcolormesh(
            this_map.lon.values,
            this_map.lat.values,
            this_map,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        if vrange:
            im.set_clim(vrange[0], vrange[1])
    ax.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree())

    if subplot_label is not None:
        plt.text(
            0, 0.95, f"({subplot_label})", transform=ax.transAxes, fontsize=fontsize["axislabels"]
        )

    # # Country borders
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth, edgecolor="white", alpha=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=linewidth*0.6, alpha=0.3)

    # Coastlines
    ax.coastlines(linewidth=linewidth, color="white", alpha=0.5)
    ax.coastlines(linewidth=linewidth * 0.6, alpha=0.3)

    if this_title:
        ax.set_title(this_title, fontsize=fontsize["titles"])
    if show_cbar:
        if cbar:
            cbar.remove()

        if bounds is not None:
            cbar = plt.colorbar(
                cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                orientation="horizontal",
                fraction=0.1,
                pad=0.02,
                spacing=cbar_spacing,
            )
        else:
            cbar = plt.colorbar(
                im,
                ax=ax,
                orientation="horizontal",
                fraction=0.1,
                pad=0.02,
                extend=extend_nonbounds,
                spacing=cbar_spacing,
            )

        deal_with_ticklabels(cbar, cbar_max, ticklabels, ticklocations, units, im)
        cbar.set_label(
            label=units,
            fontsize=fontsize["axislabels"],
            verticalalignment="center",
            labelpad=cbar_labelpad,
        )
        cbar.ax.tick_params(labelsize=fontsize["ticklabels"])
        if units is not None and "month" in units.lower():
            cbar.ax.tick_params(length=0)

    if lonlat_bin_width:
        set_ticks(lonlat_bin_width, fontsize, "y")
        # set_ticks(lonlat_bin_width, fontsize, "x")
    else:
        # Need to do this for subplot row labels
        set_ticks(-1, fontsize, "y")
        plt.yticks([])
    for x in ax.spines:
        ax.spines[x].set_visible(False)

    if show_cbar:
        return im, cbar
    else:
        return im, None


def set_ticks(lonlat_bin_width, fontsize, x_or_y):
    if x_or_y == "x":
        ticks = np.arange(-180, 181, lonlat_bin_width)
    else:
        ticks = np.arange(-60, 91, lonlat_bin_width)

    ticklabels = [str(x) for x in ticks]
    for i, x in enumerate(ticks):
        if x % 2:
            ticklabels[i] = ""

    if x_or_y == "x":
        plt.xticks(ticks, labels=ticklabels, fontsize=fontsize["ticklabels"])
    else:
        plt.yticks(ticks, labels=ticklabels, fontsize=fontsize["ticklabels"])

def deal_with_ticklabels(cbar, cbar_max, ticklabels, ticklocations, units, im):
    if ticklocations is not None:
        cbar.set_ticks(ticklocations)
        if units is not None and units.lower() == "month":
            cbar.set_ticklabels(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            )
            units == "Month"
        elif ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
    if isinstance(im, mplcol.QuadMesh):
        clim_max = im.get_clim()[1]
    else:
        clim_max = im
    if cbar_max is not None and clim_max > cbar_max:
        if ticklabels is not None:
            raise RuntimeError(
                "How to handle this now that you are specifying ticklocations separate from"
                " ticklabels?"
            )
        ticks = cbar.get_ticks()
        if ticks[-2] > cbar_max:
            raise RuntimeError(
                f"Specified cbar_max is {cbar_max} but highest bin BEGINS at {ticks[-2]}"
            )
        ticklabels = ticks.copy()
        ticklabels[-1] = cbar_max
        for i, x in enumerate(ticklabels):
            if x == int(x):
                ticklabels[i] = str(int(x))
        cbar.set_ticks(
            ticks
        )  # Calling this before set_xticklabels() avoids "UserWarning: FixedFormatter should only be used together with FixedLocator" (https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator)
        cbar.set_ticklabels(ticklabels)


def make_axis(fig, ny, nx, n):
    ax = fig.add_subplot(ny, nx, n, projection=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)  # Turn off box outline
    return ax


def equalize_colorbars(ims, center0=False, this_var=None, vrange=None):
    if vrange is None:
        vmin = np.inf
        vmax = -np.inf
        for im in ims:
            vmin = min(vmin, im.get_clim()[0])
            vmax = max(vmax, im.get_clim()[1])
    else:
        vmin = vrange[0]
        vmax = vrange[1]

    extend = "neither"
    if this_var == "HUIFRAC" and vmax > 1:
        vmax = 1
        extend = "max"

    if center0:
        v = np.max(np.abs([vmin, vmax]))
        vmin = -v
        vmax = v

    for im in ims:
        im.set_clim(vmin, vmax)

    vrange = [vmin, vmax]

    return extend, vrange
