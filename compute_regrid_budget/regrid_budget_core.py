import numpy as np
import load_data
#import netCDF4

import pandas as pd

import traceback
#import date_tools, fmon_tools, domain_tools, NK_tools, KPP_tools, watertime_tools

#import earth_constants as ec
from pathlib import Path

import argparse
#import map_divide_tools

import xarray as xr
import ECCO_helper
import regrid_tools

ERA_varnames = ["tp", "ttr",]

ECCO_varnames = [
    "THETA",
    "SALT",
    "Ue",
    "Vn",
    "dMLTdt",
    "MLT",
    "MXLDEPTH",
#    "MLG_ttl",
    "MLG_frc_sw",
    "MLG_frc_lw",
    "MLG_frc_sh",
    "MLG_frc_lh",
#    "MLG_fwf",
    "MLG_hadv",
    "MLG_vadv",
    "MLG_adv",
    "MLG_frc_dilu",
    "MLG_hdiff",
    "MLG_vdiff",
    "MLG_ent",
    "MLG_ent_wep",
    "MLG_ent_wen",
    "MLD",
    "dMLDdt",
    "dTdz_b",
    "MLU",
    "MLV",
    "MLU_g",
    "MLV_g",
    "MLU_ag",
    "MLV_ag",
    "dMLTdx",
    "dMLTdy",
    "MLHADVT_g",
    "MLHADVT_ag",
    "ENT_ADV",
    "w_b",
    "EXFempmr",
    "EXFpreci",
    "EXFevap",
    "EXFroff",
]

ECCO_3D_varnames = ["THETA", "SALT", ]

ERA5_3D_varnames = []


ERA5_short2long_mapping = {
    "ttr" : "top_net_thermal_radiation",
    "tp"  : "total_precipitation",
}

def weightedAvg(var_data, wgts):

    d = var_data.to_numpy()

    idx = np.isfinite(d)
    d = d[idx]
    w = wgts.to_numpy()[idx]

    return np.sum(d * w) / np.sum(w)

def magicalExtension(data):
    
    #data['ERA_sfc_hf']  = data['msnswrf'] + data['msnlwrf'] + data['msshf'] + data['mslhf']
    #_data['ERA_MLG_ttl_exp']  =data['ERA_sfc_hf'] / (3996*1026 *data['MLD'])
    #_data['ERA_MLG_ttl_uexp'] =data['ERA_MLG_ttl'] -data['ERA_MLG_frc']
   
    new_data = dict()
 
    new_data["dTdz_b_over_h"] = data["dTdz_b"] / data["MLD"]
    #new_data["SFCWIND"] = ( data["u10"]**2.0 + data["v10"]**2.0)**0.5
    
    new_data['MLG_residue'] = data['dMLTdt'] - (
          data['MLG_frc_sw']
        + data['MLG_frc_lw']
        + data['MLG_frc_sh']
        + data['MLG_frc_lh']
        + data['MLG_frc_dilu']
        + data['MLG_adv']
        + data['MLG_hdiff']
        + data['MLG_vdiff']
        + data['MLG_ent_wep']
        + data['MLG_ent_wen']
    )
    
    res = new_data["MLG_residue"]
    res_max = np.amax(np.abs(res[np.isfinite(res)]))
    print("Max of abs(MLG_residue): ", res_max)

    return new_data



def main(
    output_file,
    year,
    month,
    regrid_file_ERA5,
    regrid_file_ECCO,
    ERA_varnames = ERA_varnames,
    ECCO_varnames = ECCO_varnames,
):

    # Configuration

    # Need to include April and September so that
    # the climatology can be interpolated into daily data

    output_file = Path(output_file)
    output_dir = output_file.parent

    beg_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = beg_date + pd.offsets.MonthBegin()

    print("Beg: ", beg_date)
    print("End: ", end_date)

    dts = pd.date_range(beg_date, end_date, freq="D", inclusive="left")
    total_days = len(dts)

    """
    ECCO_varnames = [
        "dMLTdt",
        "MLT",
        "MXLDEPTH",
    #    "MLG_ttl",
        "MLG_frc_sw",
        "MLG_frc_lw",
        "MLG_frc_sh",
        "MLG_frc_lh",
    #    "MLG_fwf",
        "MLG_hadv",
        "MLG_vadv",
        "MLG_adv",
        "MLG_frc_dilu",
        "MLG_hdiff",
        "MLG_vdiff",
        "MLG_ent",
        "MLG_ent_wep",
        "MLG_ent_wen",
        "MLD",
        "dMLDdt",
        "dTdz_b",
        "MLU",
        "MLV",
        "MLU_g",
        "MLV_g",
        "MLU_ag",
        "MLV_ag",
        "dMLTdx",
        "dMLTdy",
        "MLHADVT_g",
        "MLHADVT_ag",
        "ENT_ADV",
        "w_b",
        "EXFempmr",
        "EXFpreci",
        "EXFevap",
        "EXFroff",
    ]
    """
    tendency_residue_tolerance = 1e-10

    domain_check_tolerance = 1e-10
    ERA_lat_raw = None
    ERA_lon_raw = None

    ecco_grid = None

    lat = None
    lon = None
    ocn_z   = None
    f_co = None

    computed_LLC_vars  = ["dTdz_b_over_h", "MLG_residue",]
    computed_ERA_vars = ["SFCWIND", ]

    all_varnames = ERA_varnames + ECCO_varnames + computed_LLC_vars + computed_ERA_vars

    # Eventually, data_good will be merge with each dataset of each box
    data_good =  np.zeros((total_days,), dtype=np.int32)

    ditch_this_month = np.nan

    print("Load regrid data")
    regrid_info_ECCO = regrid_tools.loadRegridInfo(regrid_file_ECCO)
    regrid_info_ERA5 = regrid_tools.loadRegridInfo(regrid_file_ERA5)

    data = dict()

    print("Ready to process data.")
    for d, dt in enumerate(dts):

        print("# Processing date: ", dt)
 
       

                
        I_have_all_data_for_today = True
        
        # Load ERA data
        for i, varname in enumerate(ERA_varnames):

            try:

                load_varname = varname
                long_varname = ERA5_short2long_mapping[varname]

                # Load observation (the 'truth')
                ERA5_filename = Path("dataset/ERA5-derived-daily") / "6_hourly" / long_varname / ("ERA5-derived-daily-%s-%s.nc" % (long_varname, dt.strftime("%Y-%m-%d")))

                print("Load `%s` from file: %s" % ( varname, ERA5_filename, ))
                ds_ERA = xr.open_dataset(ERA5_filename)
                _var = ds_ERA[load_varname].sel(valid_time = dt)
                var_is_3D = 'lev' in _var.dims
                _var_numpy = _var.to_numpy()
                
                if load_varname not in data:
                    if var_is_3D:
                        dim = (len(dts), _var.sizes["lev"], len(regrid_info_ECCO["lat"]), len(regrid_info_ECCO["lon"]))
                    else:
                        dim = (len(dts), len(regrid_info_ECCO["lat"]), len(regrid_info_ECCO["lon"]))
                    
                    data[load_varname] = np.zeros(dim, dtype=np.float64)

                 
                if var_is_3D:
                    pass
                else:
                    regridded_var = regrid_tools.regrid( regrid_info_ERA5, _var_numpy )
                    data[load_varname][d, :, :] = regridded_var

                #print("ERA5 shape = ", regridded_var.shape)
            except Exception as e:

                print(traceback.format_exc()) 
                print("Someting wrong happened when loading date: %s" % (dt.strftime("%Y-%m-%d"),))

                I_have_all_data_for_today = False




        ############ Loading ECCOv4 data ############

        try:

            for varname in ECCO_varnames:

                # Certain ECCO variables do not depend on mixed layer depth
                if varname in ["MXLDEPTH", "MLD", ]: 
                    ecco_filename = ECCO_helper.getECCOFilename(varname, "DAILY", dt)
                else:
                    ecco_filename = ECCO_helper.getECCOFilename(varname, "DAILY", dt)

                ecco_filename = Path("dataset") / "ECCO_LLC" / ecco_filename[0] / ecco_filename[1]

                print("Load `%s` from file: %s" % ( varname, ecco_filename, ))

                if varname == "MLD":
                    ds_ECCO = xr.open_dataset(ecco_filename).isel(time_snp=0)
                    
                else:
                    ds_ECCO = xr.open_dataset(ecco_filename).isel(time=0)

                ds_ECCO = ds_ECCO.astype(np.float64)
                _var = ds_ECCO[varname]

                # Test if 3D
                var_is_3D = 'k' in _var.dims
                _var_numpy = _var.to_numpy()
                #print(_var_numpy.shape)


                #if varname == "VVEL":
                #    _var_numpy = (_var_numpy[:, :, 1:, :] + _var_numpy[:, :, :-1, :] )  / 2

                #if varname == "UVEL":
                #    _var_numpy = (_var_numpy[:, :, :, 1:] + _var_numpy[:, :, :, :-1] )  / 2



                if var_is_3D and ocn_z is None:
                    ocn_z = _var.coords["Z"].to_numpy()


                if varname not in data:
                    if var_is_3D:
                        dim = (len(dts), _var.sizes["k"], len(regrid_info_ECCO["lat"]), len(regrid_info_ECCO["lon"]))
                    else:
                        dim = (len(dts), len(regrid_info_ECCO["lat"]), len(regrid_info_ECCO["lon"]))
                    
                    data[varname] = np.zeros(dim, dtype=np.float64)

                 
                if var_is_3D:
                    for k in range(_var.sizes['k']):
                        regridded_var = regrid_tools.regrid( regrid_info_ECCO, _var_numpy[k, :, :, :] )
                        data[varname][d, k, :, :] = regridded_var
                else:
                        regridded_var = regrid_tools.regrid( regrid_info_ECCO, _var_numpy )
                        data[varname][d, :, :] = regridded_var
                    
                #if np.all(regridded_var == 0):
                #    print("!!!!!!!!!!!!!!!!!!!!!!! all zero for varname", varname)
                
                #print("ECCO shape = ", regridded_var.shape)

        except Exception as e:

            print(traceback.format_exc()) 
            print("ECCO: Someting wrong happened when loading date: %s" % (dt.strftime("%Y-%m-%d"),))

            I_have_all_data_for_today = False


        if I_have_all_data_for_today:
            data_good[d] = 1

        else:
            data_good[d] = 0
            print("Missing data for date: ", dt)
            continue

        # Add other vairables inside

    print("Do magical extension")
    extended_data = magicalExtension(data)

    data = {**data, **extended_data}

    missing_dates = dts[data_good == 0]

    if len(missing_dates) == 0:
        
        print("Congratulations! No missing data.")

    else:

        print("Warning: Missing data.")

        for i, missing_date in enumerate(missing_dates):
            print("[%d] Missing date needed: %s" % (i, missing_date.strftime("%Y-%m-%d"),))

        missing_date_file = output_dir / ("missing_dates_%d.txt" % (year,))
        
        print("Output missing date file: %s" % (missing_date_file,))
        with open(missing_date_file, "w") as f:
            for i, missing_date in enumerate(missing_dates):
                f.write("[%d] %s\n" % (i, missing_date.strftime("%Y-%m-%d"),))


    data_vars = dict(
        data_good = (["time"], data_good),
    )

    for varname, arr in data.items():
        print(varname, " => ", arr.shape)
        dim = None
        if len(arr.shape) == 3:
            dim = ["time", "lat", "lon"]
        elif len(arr.shape) == 4:
            dim = ["time", "ocn_z", "lat", "lon"]
            
        data_vars[varname] = (dim, arr)
  
    print("lat => ", regrid_info_ERA5["lat"].shape) 
    print("lon => ", regrid_info_ERA5["lon"].shape) 
    coords = dict(
        time = (["time"], dts),
        lat = (["lat"], regrid_info_ERA5["lat"]),
        lon = (["lon"], regrid_info_ERA5["lon"]),
    )

    if ocn_z is not None: 
        coords["ocn_z"] = (["ocn_z"], ocn_z)

    output_ds = xr.Dataset(
        data_vars = data_vars,
        coords = coords,
    )

    print("Merge data_good into each dataset")
    data_good_idx = data_good == 1
    none_is_selected_idx = np.isnan(data_good) # This is an all-false boolean array. It is used to assign an entire dataset as NaN for empty boxes
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Output filename: %s" % ( output_file, ))
    output_ds.to_netcdf(
        output_file,
        unlimited_dims=["time",],
        encoding={'time': {'dtype': 'i4'}},
    )


