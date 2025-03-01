import traceback
import xarray as xr
import pandas as pd
import numpy as np
import argparse
import datetime
import os
from pathlib import Path

def computeBoxIndex(llat, llon, lat_rng, lon_rng, dlat, dlon):

    nbox_lat = int(np.floor( ( lat_rng[1] - lat_rng[0] ) / dlat))
    nbox_lon = int(np.floor( ( lon_rng[1] - lon_rng[0] ) / dlon))
 
    if nbox_lat == 0 or nbox_lon == 0:
        raise Exception("Error: The given lat lon range and spacing does not generate and box.")

    
    lat_idx = np.floor( (llat - lat_rng[0]) / dlat).astype(np.int32)
    lon_idx = np.floor( (llon - lon_rng[0]) / dlon).astype(np.int32)
    
    lat_idx[ (lat_idx >= nbox_lat) | (lat_idx < 0)] = -1    
    lon_idx[ (lon_idx >= nbox_lon) | (lon_idx < 0)] = -1  

    lat_regrid_bnds = np.linspace(lat_rng[0], lat_rng[1], nbox_lat + 1)
    lon_regrid_bnds = np.linspace(lon_rng[0], lon_rng[1], nbox_lon + 1)

    return lat_idx, lon_idx, lat_regrid_bnds, lon_regrid_bnds



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ERA5-file', type=str, help='ERA5 file that provide lat and lon.', default=None)
    parser.add_argument('--ECCO-file', type=str, help='PRISM file that provide lat and lon.', default=None)
    parser.add_argument('--output-dir', type=str, help='WRF file that provide XLAT and XLONG.', required=True)
    parser.add_argument('--lat-rng', type=float, nargs=2, help="Latitudinal range.", default=[-90.0, 90.0])
    parser.add_argument('--lon-rng', type=float, nargs=2, help="Latitudinal range.", default=[0.0, 360.0])
    parser.add_argument('--dlat', type=float, help="dlat in latitudinal direction.", required=True)
    parser.add_argument('--dlon', type=float, help="dlon in longitudinal direction.", required=True)
    args = parser.parse_args()

    print(args)

    if_has_file = { model : getattr(args, "%s_file" % model) is not None for model in ["ERA5", "ECCO"] }

    if np.all([ not has_file for _, has_file in if_has_file.items() ] ):
        raise Exception("Must provide either ERA5 or ECCO file")


    max_lat_idx = int(np.floor( ( args.lat_rng[1] - args.lat_rng[0] ) / args.dlat))
    max_lon_idx = int(np.floor( ( args.lon_rng[1] - args.lon_rng[0] ) / args.dlon))
    
    print("max_lat_idx: ", max_lat_idx)
    print("max_lon_idx: ", max_lon_idx)
   

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if if_has_file["ERA5"]:
        ERA5_ds = xr.open_dataset(args.ERA5_file)
       
        ERA5_lat = ERA5_ds.coords["latitude"].to_numpy()
        ERA5_lon = ERA5_ds.coords["longitude"].to_numpy() % 360.0
        ERA5_llat, ERA5_llon = np.meshgrid(ERA5_lat, ERA5_lon, indexing='ij')

        ERA5_lat_idx, ERA5_lon_idx, lat_regrid_bnds, lon_regrid_bnds = computeBoxIndex(ERA5_llat, ERA5_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)

        ERA5_regrid_ds = xr.Dataset(
            data_vars = dict(
                lat_idx = (["lat", "lon"], ERA5_lat_idx),
                lon_idx = (["lat", "lon"], ERA5_lon_idx),
                lat_regrid_bnd = (["lat_regrid_bnd",], lat_regrid_bnds),
                lon_regrid_bnd = (["lon_regrid_bnd",], lon_regrid_bnds),
            ),
            coords = dict(
                lat = (["lat"], ERA5_lat),
                lon = (["lon"], ERA5_lon),
            ),
            attrs = dict(
                nlat_box = max_lat_idx+1,
                nlon_box = max_lon_idx+1,
            )
        )

        output_file = output_dir / "mask_ERA5.nc"
        print("Output file: ", output_file)
        ERA5_regrid_ds.to_netcdf(output_file)

    if if_has_file["ECCO"]:
        
        ECCO_ds = xr.open_dataset(args.ECCO_file)
        ECCO_llat = ECCO_ds["YC"].to_numpy()
        ECCO_llon = ECCO_ds["XC"].to_numpy() % 360.0
        ECCO_lat_idx, ECCO_lon_idx, _, _ = computeBoxIndex(ECCO_llat, ECCO_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)
        
        print(ECCO_llat.shape)

        ECCO_regrid_ds = xr.Dataset(
            data_vars = dict(
                lat_idx = (["tile", "lat", "lon"], ECCO_lat_idx),
                lon_idx = (["tile", "lat", "lon"], ECCO_lon_idx),
                lat_regrid_bnd = (["lat_regrid_bnd",], lat_regrid_bnds),
                lon_regrid_bnd = (["lon_regrid_bnd",], lon_regrid_bnds),
            ),
            coords = dict(
                llat = (["tile", "lat", "lon"], ECCO_llat),
                llon = (["tile", "lat", "lon"], ECCO_llon),
            ),
            attrs = dict(
                nlat_box = max_lat_idx+1,
                nlon_box = max_lon_idx+1,
            )
        )

        output_file = output_dir / "mask_ECCO.nc"
        print("Output file: ", output_file)
        ECCO_regrid_ds.to_netcdf(output_file)
