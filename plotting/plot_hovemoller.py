import numpy as np
import xarray as xr
import pandas as pd

import traceback
from pathlib import Path
import argparse
from datetime import (timedelta, datetime, timezone)
import cmocean
import tool_fig_config

def getIfExists(d, k, default=None):

    result = None

    if k in d:
        result = d[k]
    else:
        result = default

    return result
        

plot_infos = dict(
    ttr = dict(
        label = "OLR",
        unit = "$\\mathrm{W} \\cdot \\mathrm{m}^{-2}$",
        levs = np.linspace(-1, 1, 21) * 50,
        factor = - 3600,
    ),

    MLT = dict(
        label = "SST",
        unit = "$\\mathrm{K}$",
        levs = np.linspace(-1, 1, 21) * 0.5,
        factor = 1,
    ),

    dMLTdt = dict(
        label = "$\\partial \\mathrm{SST} / \\partial t$",
        unit = "$1 \\times 10^{-6} \\mathrm{K}\\cdot\\mathrm{s}^{-1}$",
        levs = np.linspace(-1, 1, 21) * 1,
        factor = 1e-6,
    ),

    EXFpreci = dict(
        label = "Precipitation",
        unit = "$1 \\mathrm{mm} \\cdot \\mathrm{day}^{-1}$",
        levs = np.linspace(-1, 1, 21) * 5,
        factor = 1e-3 / 86400,
        
        ttl = dict(
            levs = np.linspace(0, 1, 21) * 30,
            cmap = "cmo.rain",
        ),
    ),


)

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-file', type=str, help='Input file', required=True)
parser.add_argument('--varname', type=str, required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--filtered-type', type=str, help='Output file', choices=["anomalies", "bandpass", "ttl"], required=True)

parser.add_argument('--show-title', action="store_true")
parser.add_argument('--show-numbering', action="store_true")
parser.add_argument('--numbering', type=int, default=1)
parser.add_argument('--numbering-list', type=str, default="abcdefghijklmn")

parser.add_argument('--time-rng', type=str, nargs=2, help='Output file')
parser.add_argument('--lon-rng', type=float, nargs=2, help='Output file', default=None)
parser.add_argument('--lat-rng', type=float, nargs=2, help='Output file')
parser.add_argument('--aspect-ratio', type=float, help='Height / width', default=2.0)
parser.add_argument('--no-display', action="store_true")
parser.add_argument('--reverse-time', action="store_true")

args = parser.parse_args()
print(args)

time_beg = pd.Timestamp(args.time_rng[0])
time_end = pd.Timestamp(args.time_rng[1])

selected_dts = pd.date_range(time_beg, time_end, freq="D", inclusive="both")

da = xr.open_dataset(args.input_file)[args.varname]
da = da.sel(time=selected_dts)

da = da.where(
    (da.coords["lat"] > args.lat_rng[0]) &
    (da.coords["lat"] < args.lat_rng[1])
).mean(dim="lat", skipna=True)



plot_info = plot_infos[args.varname]

# Plot data
print("Loading Matplotlib...")
import matplotlib as mpl
if args.no_display is False:
    mpl.use('TkAgg')
else:
    mpl.use('Agg')
    mpl.rc('font', size=15)
    mpl.rc('axes', labelsize=15)
     
 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cmocean as cmo

print("done")

ncol = 1
nrow = 1

w = 4.0
h = w * args.aspect_ratio

figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = h,
    wspace = 1.0,
    hspace = 0.5,
    w_left = 1.0,
    w_right = 1.5,
    h_bottom = 0.7,
    h_top = 0.5,
    ncol = ncol,
    nrow = nrow,
)


fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

cmap = cmocean.cm.balance

ax_flatten = ax.flatten()
#cmap.set_over("green")
#cmap.set_under("yellow")
factor = getIfExists(plot_info, "factor", 1)


if args.filtered_type == "ttl":
    levs = getIfExists(plot_info["ttl"], "levs")
    cmap = getIfExists(plot_info["ttl"], "cmap")
else:
    levs = getIfExists(plot_info, "levs")
    cmap = getIfExists(plot_info, "cmap", "bwr")
   

_ax = ax_flatten[0]
coords = da.coords


_d = da.to_numpy() / factor
mappable = _ax.contourf(
    coords["lon"], coords["time"],
    _d,
    levels=levs,
    cmap=cmap,
    extend="both",
)

cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)

cb.ax.set_ylabel("%s [%s]" % (plot_info["label"], plot_info["unit"]), size=15)

if args.lon_rng is not None:
    _ax.set_xlim(args.lon_rng)

lon_rng = _ax.get_xlim()
print("lon_rng = ", lon_rng)


spacing = 60
first_tick = np.ceil(lon_rng[0] / spacing) * spacing 
last_tick  = np.floor(lon_rng[1] / spacing) * spacing 
_ax.set_xticks(np.arange(first_tick, last_tick+spacing, spacing))

if args.reverse_time:
    
    _ax.invert_yaxis()

_ax.grid()    
_ax.set_xlabel("Longitude [degree east]")

if args.show_title:
    
    title = "%s" % plot_info["label"]
    if args.show_numbering:
        title = "(%s) %s" % (
            args.numbering_list[args.numbering-1],
            title,
        )

    _ax.set_title(title)


if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


print("Finished.")
