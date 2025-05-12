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
 

parser = argparse.ArgumentParser(
                    prog = 'plot_skill',
                    description = 'Plot prediction skill of GFS on AR.',
)

parser.add_argument('--input-dir', type=str, help='Input file', required=True)
parser.add_argument('--output', type=str, help='Output file', default="")
parser.add_argument('--varnames', type=str, nargs="+", help='Output file', default=["dMLTdt", "MLG_frc", "MLG_nonfrc"])
parser.add_argument('--ncol', type=int, help='Output file', default=4)
parser.add_argument('--time-rng', type=str, nargs=2, help='Output file')
parser.add_argument('--lon-rng', type=float, nargs=2, help='Output file')
parser.add_argument('--lat-rng', type=float, nargs=2, help='Output file')
parser.add_argument('--thumbnail-offset', type=int, help='Output file', default=0)
parser.add_argument('--add-thumbnail-title', action="store_true")
parser.add_argument('--no-display', action="store_true")

args = parser.parse_args()
print(args)

input_dir = Path(args.input_dir)

time_beg = pd.Timestamp(args.time_rng[0])
time_end = pd.Timestamp(args.time_rng[1])

selected_dts = pd.date_range(time_beg, time_end, freq="D", inclusive="both")

# Load files
data = dict()

for varname in args.varnames:
    
    if varname == "BLANK":
        continue
    
    print("Loading varname: ", varname)
    filename = input_dir / f"bandpass-lanczos_{varname:s}.nc"
    da = xr.open_dataset(filename)[varname]
    da = da.sel(time=selected_dts).mean(dim="time")

    data[varname] = da

G_scale  = np.linspace(-1, 1, 17) * 0.4
G_scale_breakdown  = np.linspace(-1, 1, 17) * 0.3
G_scale3 = np.linspace(-1, 1, 11) * 0.1

plot_infos = {
 
    "sst" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "SST",
        "unit"  : "$ \\mathrm{K} $",
        "factor" : 1,
    },


    "msl" : {
        "levs": np.linspace(-1, 1, 17) * 8,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "SLP",
        "unit"  : "$ \\mathrm{hPa} $",
        "factor" : 100.0,
    },


    "MLT" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "MLT",
        "unit"  : "$ \\mathrm{K} $",
        "factor" : 1,
    },

   
    "dMLTdt" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{loc}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    },

    "MLG_frc" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{sfc}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_nonfrc" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{ocn}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_vdiff" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{vdiff}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_vdiff_entwep" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{ent}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_hdiff" : {
        "levs": G_scale3,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{hdif}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_vmix" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{vmix}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_vmixall" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{vmix}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 



    "MLG_ent_wen" : {
        "levs": G_scale3,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{det}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 


    "MLG_ent" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{ent}} + \\dot{\\overline{\\Theta}}_{\\mathrm{det}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_adv" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\mathrm{adv}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

 
    "MLG_frc_sw" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{sw}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_lw" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{lw}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_sh" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{sen}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_lh" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{lat}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLG_frc_dilu" : {
        "levs": G_scale3,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "$ \\dot{\\overline{\\Theta}}_{\\mathrm{dilu}} $",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 


    "MLHADVT_g" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "Geo HADV",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "MLHADVT_ag" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "Ageo HADV",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 

    "ENT_ADV" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 2, 11),
        "label" : "Ent ADV",
        "unit"  : "$ 10^{-6} \\, \\mathrm{K} / \\mathrm{s} $",
        "factor" : 1e-6,
    }, 



    "MXLDEPTH" : {
        "levs": np.linspace(-1, 1, 11) * 20,
        "levs_std": np.linspace(0, 20, 21),
        "label" : "MXLDEPTH",
        "unit"  : "$\\mathrm{m}$",
        "factor" : 1.0,
    }, 
 
    "MLD" : {
        "levs": np.linspace(-1, 1, 11) * 20,
        "levs_std": np.linspace(0, 100, 21),
        "label" : "MLD",
        "unit"  : "$\\mathrm{m}$",
        "factor" : 1.0,
    }, 


    "IWV" : {
#        "levs": np.linspace(0, 1, 11) * 50,
        "levs": np.linspace(0, 1, 11) * 10,
        "levs_std": np.linspace(0, 20, 21),
        "label" : "IWV",
        "unit"  : "$\\mathrm{kg} \\, / \\, \\mathrm{m}^2$",
        "factor" : 1.0,
    }, 

    "IVT" : {
#        "levs": np.linspace(0, 1, 13) * 600,
        "levs": np.linspace(0, 1, 11) * 300,
        "levs_std": np.linspace(0, 1, 11) * 200,
        "label" : "IVT",
        "unit"  : "$\\mathrm{kg} \\, / \\, \\mathrm{m} \\, / \\, \\mathrm{s}$",
        "factor" : 1.0,
    }, 

    "SFCWIND" : {
        "levs": np.linspace(-1, 1, 13) * 6,
        "levs_std": np.linspace(0, 5, 11),
        "label" : "$\\left| \\vec{V}_{10} \\right|$",
        "unit"  : "$\\mathrm{m} \\, / \\, \\mathrm{s}$",
        "factor" : 1.0,
    }, 

    "EXFpreci" : {
        "levs": np.linspace(-1, 1, 11) * 10,
        "levs_std": np.linspace(0, 1, 11) * 10,
        "label" : "Precip",
        "unit"  : "$\\mathrm{mm} \\, / \\, \\mathrm{day}$",
        "factor" : 1.0 / (86400.0 * 1e3),
    }, 

    "EXFempmr" : {
        "levs": np.linspace(-1, 1, 11) * 10,
        "levs_std": np.linspace(0, 1, 11) * 10,
        "label" : "E - P - R",
        "unit"  : "$\\mathrm{mm} \\, / \\, \\mathrm{day}$",
        "factor" : - 1.0 / (86400.0 * 1e3),
    }, 

    "EXFevap" : {
        "levs": G_scale,
        "levs_std": np.linspace(0, 1, 11) * 1,
        "label" : "Evap heat flux",
        "unit"  : "$\\mathrm{W} \\, / \\, \\mathrm{m}^2$",
        "factor" : - 1e-6 / 2.5e6 * (3996*50),
    }, 




    "tcc" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 1, 11) * 0.5,
        "label" : "TCC",
        "unit"  : "None",
        "factor" : 1.0,
    }, 


    "lcc" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 1, 11) * 0.5,
        "label" : "LCC",
        "unit"  : "None",
        "factor" : 1.0,
    }, 

    "mcc" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 1, 11) * 0.5,
        "label" : "MCC",
        "unit"  : "None",
        "factor" : 1.0,
    }, 

    "hcc" : {
        "levs": np.linspace(-1, 1, 11) * 0.5,
        "levs_std": np.linspace(0, 1, 11) * 0.5,
        "label" : "HCC",
        "unit"  : "None",
        "factor" : 1.0,
    }, 


    "dTdz_b" : {
        "levs": np.linspace(-1, 1, 11) * 2,
        "levs_std": np.linspace(0, 1, 11) * 2,
        "label" : "$\\partial \\Theta_{\\eta - h} / \\partial z $",
        "unit"  : "$10^{-2} \\, \\mathrm{K} \\, / \\, \\mathrm{m}$",
        "factor" : 1e-2,
    }, 


}


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
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


print("done")

cent_lon = 180.0

plot_lon_l = args.lon_rng[0] % 360
plot_lon_r = args.lon_rng[1] % 360
plot_lat_b = args.lat_rng[0]
plot_lat_t = args.lat_rng[1]

proj = ccrs.PlateCarree(central_longitude=cent_lon)
proj_norm = ccrs.PlateCarree()

varnames = args.varnames


ncol = args.ncol
nrow = int(np.ceil(len(varnames)/ncol))


w = 6
h = w * (plot_lat_t - plot_lat_b) / (plot_lon_r - plot_lon_l)


figsize, gridspec_kw = tool_fig_config.calFigParams(
    w = w,
    h = h,
    wspace = 2.0,
    hspace = 1.0,
    w_left = 1.0,
    w_right = 2.2,
    h_bottom = 1.0,
    h_top = 1.0,
    ncol = ncol,
    nrow = nrow,
)


fig, ax = plt.subplots(
    nrow, ncol,
    figsize=figsize,
    subplot_kw=dict(projection=proj, aspect="auto"),
    gridspec_kw=gridspec_kw,
    constrained_layout=False,
    squeeze=False,
)

ax_flatten = ax.flatten()

coords = data[varnames[0]].coords
cmap = cmocean.cm.balance

mappables = [ None for i in range(len(varnames)) ]

thumbnail_cnt = 0

for i, varname in enumerate(varnames):

    _ax = ax_flatten[i]
    if varname == "BLANK":
        fig.delaxes(_ax)
        continue

    plot_info = plot_infos[varname]

    da = data[varname]
    # Set title for different month except for the whole average mon==7
    #_ax.set_title("(%s) %s" % ("abcdefghijklmnopqrstu"[i], plot_info["label"]), size=30)
    factor = getIfExists(plot_info, "factor", 1)

    _d = da.to_numpy() / factor
    mappable = _ax.contourf(
        coords["lon"], coords["lat"],
        _d,
        levels=plot_info["levs"],
        cmap="bwr",
        extend="both",
        transform=proj_norm,
    )

    if args.add_thumbnail_title :
        print("THUMBNAIL_CNT = %d" % (thumbnail_cnt,))
        _ax.set_title("(%s) %s" % ("abcdefghijklmnopqrstuvwxyz"[thumbnail_cnt + args.thumbnail_offset], plot_info["label"]), size=30)
        thumbnail_cnt += 1
            

    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.03, spacing=0.05)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)

    unit_str = getIfExists(plot_info, "unit", "")
    cb.ax.set_ylabel("%s [%s]" % (plot_info["label"], unit_str), size=15)
   
    _ax.set_global()
    #_ax.gridlines()
    _ax.coastlines(color='gray')
    _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=proj_norm)

    gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')

    gl.xlabels_top   = False
    gl.ylabels_right = False

    #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
    #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'black'}
    gl.ylabel_style = {'size': 12, 'color': 'black'}



if not args.no_display:
    plt.show()

if args.output != "":

    print("Saving output: ", args.output)    
    fig.savefig(args.output, dpi=200)


print("Finished.")
