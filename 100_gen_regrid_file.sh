#!/bin/bash

source 00_setup.sh



ECCO_file=dataset/ECCO_LLC/ECCO_L4_TEMP_SALINITY_LLC0090GRID_DAILY_V4R4/OCEAN_TEMPERATURE_SALINITY_day_mean_2017-05-01_ECCO_V4r4_native_llc0090.nc
ERA5_file=dataset/ERA5-derived-daily/6_hourly/top_net_thermal_radiation/ERA5-derived-daily-top_net_thermal_radiation-1993-01-30.nc


python3 compute_regrid_budget/gen_regrid_file.py \
    --ECCO-file $ECCO_file   \
    --ERA5-file $ERA5_file   \
    --lat-rng   -31 31       \
    --lon-rng   100 260      \
    --dlat 2                 \
    --dlon 2                 \
    --output-dir $gendata_dir/regrid_files
