import numpy as np
import load_data
import netCDF4

import pandas as pd

import traceback
import date_tools, fmon_tools, domain_tools, NK_tools, KPP_tools, watertime_tools

import earth_constants as ec
from pathlib import Path

import argparse
import map_divide_tools

import xarray as xr
import ECCO_helper

ERA_varnames = ["tthr",]

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
 
    new_data["dTdz_b_over_h"] =data["dTdz_b"] /data["MLD"]
    new_data["SFCWIND"] = (_data["u10"]**2.0 +data["v10"]**2.0)**0.5
    
    new_data['MLG_residue'] =data['dMLTdt'] - (
         data['MLG_frc_sw']
        +data['MLG_frc_lw']
        +data['MLG_frc_sh']
        +data['MLG_frc_lh']
        +data['MLG_frc_dilu']
        +data['MLG_adv']
        +data['MLG_hdiff']
        +data['MLG_vdiff']
        +data['MLG_ent_wep']
        +data['MLG_ent_wen']
    )
    
    res = new_data["MLG_residue"].to_numpy()
    res_max = np.amax(np.abs(res[np.isfinite(res)]))
    print("Max of abs(MLG_residue): ", res_max)

    return new_data



def constructTimeseries(
    output_dir,
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


    beg_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = beg_date + pd.offsets.offsets.DateOffset(months=1)

    print("Beg: ", beg_date)
    print("End: ", end_date)

    dts = pd.date_range(beg_date, end_date, freq="D", inclusive="both")
    total_days = len(dts)

    lat_rng = np.array(lat_rng)
    lon_rng = np.array(lon_rng) % 360

    lon_bnds = np.array([ lon_rng[0] + (lon_rng[1] - lon_rng[0]) / lon_nbox * i for i in range(lon_nbox+1)])
    lat_bnds = np.array([ lat_rng[0] + (lat_rng[1] - lat_rng[0]) / lat_nbox * i for i in range(lat_nbox+1)])

    boxes = map_divide_tools.makeDividedBoxes(lon_bnds, lat_bnds)
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

    tendency_residue_tolerance = 1e-10

    domain_check_tolerance = 1e-10
    ERA_lat_raw = None
    ERA_lon_raw = None

    ecco_grid = None

    lat = None
    lon = None
    f_co = None

    computed_LLC_vars  = ["dTdz_b_over_h", "MLG_residue",]
    computed_ERA_vars = ["SFCWIND", ]

    all_varnames = ERA_varnames + ECCO_varnames + computed_LLC_vars + computed_ERA_vars

    print("Construct output datasets: ts_datasets")
    ts_datasets = [
        xr.Dataset(
            data_vars = { 
                varname : (['time',], np.zeros((total_days,), dtype=np.float64)) 
                for varname in all_varnames 
            },

            coords = {
                'time' : t_vec_npdatetime,
            },
        ) 
        for b in range(len(boxes))
    ]

    print("Construct output datasets: full_dataset")

    full_dataset = xr.Dataset(
        data_vars = { 
            varname : (['time', 'lat', 'lon', ], np.zeros((total_days, len(lat_bnds)-1, len(lon_bnds)-1), dtype=np.float64)) 
            for varname in all_varnames
        },
        coords = {
            'time' : t_vec_npdatetime,
            'lat'  : (lat_bnds[:-1] + lat_bnds[1:]) / 2,
            'lon'  : (lon_bnds[:-1] + lon_bnds[1:]) / 2,
        },
    ) 

    box_number = np.zeros((len(lat_bnds)-1, len(lon_bnds)-1), dtype=np.int32)

    for b, box in enumerate(boxes):
        j = box["j"]
        i = box["i"]
        box_number[j, i] = box["n"]

    full_dataset = xr.merge([
        
        full_dataset,
        xr.Dataset(
            { 
                "box_number" : (['lat', 'lon'], box_number),
            },
            coords = {
                'lat'  : (lat_bnds[:-1] + lat_bnds[1:]) / 2,
                'lon'  : (lon_bnds[:-1] + lon_bnds[1:]) / 2,
                'lat_bnds' : lat_bnds,
                'lon_bnds' : lon_bnds,
            },
        ),
    ])


    full_dataset.coords["time"].encoding["units"] = "days since 1990-01-01"

    # Eventually, data_good will be merge with each dataset of each box
    data_good = xr.DataArray(
        name = "data_good",
        data =  np.zeros((total_days,), dtype=np.int32),
        dims=["time",],
        coords=dict(time=dts),
    )

    ditch_this_month = np.nan

    print("Ready to process data.")

    for d, dt in enumerate(dts):

        print("# Processing date: ", dt)
            
        _data = {}
                
        I_have_all_data_for_today = True
        
        # Load ERA data
        for i, varname in enumerate(ERA_varnames):

            try:

                load_varname = varname

                # Load observation (the 'truth')
                info = load_data.getFileAndIndex(ERA_type, _t, root_dir="data", varname=varname)

                print("Load `%s` from file: %s" % ( varname, info['filename'] ))
                ds_ERA = xr.open_dataset(info["filename"])
                _var = ds_ERA[varname].isel(time=0)

                if ERA_lat_raw is None:
                  
                    print("Coordinate loading...")

                    mask_ERA = xr.open_dataset(args.mask_ERA)["mask"].to_numpy()
                   
                    lat_name = ds_ERA[varname].dims[1]
                    lon_name = ds_ERA[varname].dims[2]

                    # For some reason, xarray save latitude, longitude
                    # into variables lat and lon. This might be a bug
                    #lat_name = "lat"
                    #lon_name = "lon"

                    if not (lat_name in ["lat", "latitude"]):
                        raise Exception("Unknown latitude name: %s" % (lat_name,))

                    if not (lon_name in ["lon", "longitude"]):
                        raise Exception("Unknown longitude name: %s" % (lat_name,))

                    ERA_lat_raw = ds_ERA.coords[lat_name]
                    ERA_lon_raw = ds_ERA.coords[lon_name] % 360

                    print("ERA_lat_raw = ", ERA_lat_raw)
                    print("ERA_lon_raw = ", ERA_lon_raw)

                    ERA_lat, ERA_lon = np.meshgrid(ERA_lat_raw.to_numpy(), ERA_lon_raw.to_numpy(), indexing='ij')

                    ERA_wgts = np.cos(ERA_lat * np.pi / 180)
                    ERA_grid = xr.Dataset(
                        { 
                            "llat" : (['lat', 'lon'], ERA_lat), 
                            "llon" : (['lat', 'lon'], ERA_lon), 
                            "wgts" : (['lat', 'lon'], ERA_wgts),
                        },

                        coords = {
                            'lat' : ERA_lat_raw,
                            'lon' : ERA_lon_raw,
                        },
                    )

                    # Create label 
                    for b, box in enumerate(boxes):
                        
                        box['ERA_subset_idx'] = (
                              ( ERA_lat >= box['polygon']['lat_bnds'][0] )
                            & ( ERA_lat <  box['polygon']['lat_bnds'][1] )
                            & ( ERA_lon >= box['polygon']['lon_bnds'][0] )
                            & ( ERA_lon <  box['polygon']['lon_bnds'][1] )
                            & ( mask_ERA == 1)
                        )

                        box['empty_ERA'] = np.sum(box['ERA_subset_idx']) == 0

                        if box['empty_ERA']:
                            if args.ignore_empty_box:
                                print("[ERA] Ignore empty box: %d" % (b,))
                            else:
                                raise Exception("ERROR: No point is selected in ERA latlon grid.")

                        #box['ERA_wgts'] = ERA_grid.wgts.where(box['ERA_subset_idx'], other=0.0).rename("ERA_wgts")
                        box['ERA_wgts'] = ERA_grid.wgts.to_numpy()[box['ERA_subset_idx']]


                # Subset after
                #_var = _var.where(ERA_subset_idx)

                _data[load_varname] = _var.load()

            except Exception as e:

                print(traceback.format_exc()) 
                print("Someting wrong happened when loading date: %s" % (_t.strftime("%Y-%m-%d"),))

                I_have_all_data_for_today = False




        ############ Loading ECCOv4 data ############

        try:

            for varname in ECCO_varnames:

                # Certain ECCO variables do not depend on mixed layer depth
                if varname in ["MXLDEPTH", "MLD", ]: 
                    ecco_filename = ECCO_helper.getECCOFilename(varname, "DAILY", _t)
                else:
                    ecco_filename = ECCO_helper.getECCOFilename(varname, "DAILY", _t)

                ecco_filename = "data/ECCO_LLC/%s/%s" % ecco_filename

                print("Load `%s` from file: %s" % ( varname, ecco_filename, ))

                if varname == "MLD":
                    ds_ECCO = xr.open_dataset(ecco_filename).isel(time_snp=0)
                    
                else:
                    ds_ECCO = xr.open_dataset(ecco_filename).isel(time=0)

                ds_ECCO = ds_ECCO.astype(np.float64)

                if ecco_grid is None:
                   
                    print("`ecco_grid` is None. Generating it and related box information.") 
                    mask_ECCO = xr.open_dataset(args.mask_ECCO).mask.to_numpy()

                    ecco_grid = ECCO_helper.getECCOGrid()

                    ecco_lat = ecco_grid["YC"].to_numpy()
                    ecco_lon = ecco_grid["XC"].to_numpy() % 360

                    print("ecco_lat: ", ecco_lat)
                    print("ecco_lon: ", ecco_lon)

                    for b, box in enumerate(boxes):
                    
                        poly = box['polygon']

                        box['ecco_subset_idx'] = (
                            (ecco_lat   >= poly['lat_bnds'][0])
                            & (ecco_lat <  poly['lat_bnds'][1])
                            & (ecco_lon >= poly['lon_bnds'][0])
                            & (ecco_lon <  poly['lon_bnds'][1])
                            & (mask_ECCO == 1)
                        )
                        
                        box['empty_ecco'] = np.sum(box['ecco_subset_idx']) == 0

                        if box['empty_ecco']:
                            if args.ignore_empty_box:
                                print("[ECCO] Ignore empty box: %d" % (b,))
                            else:
                                raise Exception("ERROR: No point is selected in ECCO LLC grid.")
                        
                        #box['ecco_wgts'] = ecco_grid.rA.where(box['ecco_subset_idx'], other=0.0)
                        box['ecco_wgts'] = ecco_grid.rA.to_numpy()[box['ecco_subset_idx']]


                # subset afterwards   
                #ds_ECCO = ds_ECCO.where(ecco_subset_idx)
                _data[varname] = ds_ECCO[varname].load()

        except Exception as e:

            print(traceback.format_exc()) 
            print("ECCO: Someting wrong happened when loading date: %s" % (_t.strftime("%Y-%m-%d"),))

            I_have_all_data_for_today = False


        if I_have_all_data_for_today:
            data_good[d] = 1

        else:
            data_good[d] = 0
            print("Missing data for date: ", _t)
            continue

        # Add other vairables inside
        print("Do magical extension")
        magicalExtension(_data)


        print("Do average of each box")
        # Make average of each box
        for varname, var_data in _data.items():
            #print("varname: ", varname)
            for b, box in enumerate(boxes):

                if box['empty_ecco'] or box['empty_ERA']:
                    continue

                ts_ds = ts_datasets[b]
                if (varname in ERA_varnames) or (varname in computed_ERA_vars):
                    subset_idx_varname  = 'ERA_subset_idx'
                    subset_wgts_varname = 'ERA_wgts'
                    
                elif (varname in ECCO_varnames) or (varname in computed_LLC_vars):
                    subset_idx_varname  = 'ecco_subset_idx'
                    subset_wgts_varname = 'ecco_wgts'

                else:
                    raise Exception("Unknown variable : %s" % (varname,) )

                #_masked_data = var_data.where(box[subset_idx_varname])
                #_wgts        = box[subset_wgts_varname]  # already subsetted
                #ts_ds[varname][d] = weightedAvg(_masked_data, _wgts)
     
                idx = box[subset_idx_varname]
                _masked_data = var_data.to_numpy()[idx]
                _wgts        = box[subset_wgts_varname]  # already subsetted
                ts_ds[varname][d] = np.sum(_masked_data * _wgts) / np.sum(_wgts)
                


                
        print("Do recheck of MLG budget")
        for b in range(len(boxes)):

            if box['empty_ecco'] or box['empty_ERA']:
                continue

            ts_ds = ts_datasets[b]

     
            MLG_recheck = (ts_ds['dMLTdt'][d] - (
                  ts_ds['MLG_frc_sw'][d]
                + ts_ds['MLG_frc_lw'][d]
                + ts_ds['MLG_frc_sh'][d]
                + ts_ds['MLG_frc_lh'][d]
                + ts_ds['MLG_frc_dilu'][d]
                + ts_ds['MLG_adv'][d]
                + ts_ds["MLG_vdiff"][d]
                + ts_ds["MLG_hdiff"][d]
                + ts_ds["MLG_ent_wep"][d]
                + ts_ds["MLG_ent_wen"][d]
            )).rename('MLG_recheck')

            print("[box=%d][v=%s] Double check residue using averaged values: " % (b, varname), ts_ds["MLG_residue"].data[d])



    data_good_t = t_vec_npdatetime[ data_good == 1 ]
    missing_dates = date_tools.findMissingDatetime(data_good_t, beg_date, end_date, timedelta(days=1))

    if len(missing_dates) == 0:
        
        print("Congratulations! No missing data.")

    else:

        print("Warning: Missing data.")


        for i, missing_date in enumerate(missing_dates):
            print("[%d] Missing date needed: %s" % (i, missing_date.strftime("%Y-%m-%d"),))

        missing_date_file = "%s/missing_dates_%d.txt" % (args.output_dir, args.wateryear)
        print("Output missing date file: %s" % (missing_date_file,))
        with open(missing_date_file, "w") as f:
            for i, missing_date in enumerate(missing_dates):
                f.write("[%d] %s\n" % (i, missing_date.strftime("%Y-%m-%d"),))

    print("Merge data_good into each dataset")
    data_good_idx = data_good == 1
    none_is_selected_idx = np.isnan(data_good) # This is an all-false boolean array. It is used to assign an entire dataset as NaN for empty boxes

    if ecco_grid is None or ERA_lat_raw is None:
        
        raise Exception("ERROR: Grid information is not loaded (ecco_grid and ERA_lat_raw).")


    for i, ts_ds in enumerate(ts_datasets):

        if boxes[i]['empty_ERA'] or boxes[i]['empty_ecco']:
            ts_datasets[i] = ts_ds.where(none_is_selected_idx).merge(data_good)
        else: 
            ts_datasets[i] = ts_ds.where(data_good_idx).merge(data_good)

    print("Merge each timeseries into a complete file.")
    for b, box in enumerate(boxes):
        
        i = box["i"]
        j = box["j"]
        for varname in all_varnames:
            full_dataset[varname][:, j, i] = ts_datasets[b][varname]

    if args.output_dir != "":
                
        print("Output directory: %s" % (args.output_dir,))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
       
        output_filename_full_dataset = "%s/%s" % (args.output_dir, args.output_filename)
            
        print("Output filename: %s" % ( output_filename_full_dataset, ))
        full_dataset.to_netcdf(
            output_filename_full_dataset,
            unlimited_dims=["time",],
            encoding={'time': {'dtype': 'i4'}},
        )
     


print("Loading libraries completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        prog = 'plot_skill',
                        description = 'Plot prediction skill of GFS on AR.',
    )

    parser.add_argument('--wateryear',   type=int, help='Date string: yyyy-mm-dd', required=True)
    parser.add_argument('--output-dir', type=str, help='Output directory', default="")
    parser.add_argument('--output-filename', type=str, help='Output filename', default="full_dataset.nc")
    parser.add_argument('--lat-rng',    type=float, nargs=2, help='Latitude  range', required=True)
    parser.add_argument('--lon-rng',    type=float, nargs=2, help='Longitude range. 0-360', required=True)
    parser.add_argument('--lat-nbox',   type=int, help='Latitude  range', required=True)
    parser.add_argument('--lon-nbox',   type=int, help='Longitude range. 0-360', required=True)
    parser.add_argument('--mask-ERA',  type=str, help='mask file of ERA', required=True)
    parser.add_argument('--ERA-type',  type=str, help='Options: ERA5 or ERAInterim', required=True, choices=["ERA5", "ERAInterim"])
    parser.add_argument('--mask-ECCO',  type=str, help='mask file of ECCO', required=True)
    parser.add_argument('--ignore-empty-box',  action="store_true")

    args = parser.parse_args()

    print(args)



