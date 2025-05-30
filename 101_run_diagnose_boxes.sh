#!/bin/bash

source 000_setup.sh
source tools/pretty_latlon.sh

nproc=5


output_root=$gendata_dir/regrid

beg_year=1993
end_year=2016

# format: lat_m lat_M lon_m lon_M lat_nbox lon_nbox
spatial_rngs=(
    -31 31 100 -100 31 80
)

nparms=6


mask_ERA5=$gendata_dir/mask/mask_ERA5.nc
mask_ECCO=$gendata_dir/mask/mask_ECCO.nc

regrid_file_ERA5=$gendata_dir/regrid_files/regrid_ERA5.nc
regrid_file_ECCO=$gendata_dir/regrid_files/regrid_ECCO.nc

mkdir -p mask

if [ ! -f "$mask_ECCO" ]; then
    echo "Mask file $mask_ECCO does not exist. Generating now..."
    #python3 make_mask_ECCO.py
fi

if [ ! -f "$mask_ERAinterim" ]; then
    echo "Mask file $mask_ERA5 does not exist. Generating now..."
    #python3 make_mask_ERA5.py
fi

for i in $( seq 1 $(( ${#spatial_rngs[@]} / $nparms )) ); do

    lat_min=${spatial_rngs[$(( ( i - 1 ) * $nparms + 0 ))]}
    lat_max=${spatial_rngs[$(( ( i - 1 ) * $nparms + 1 ))]}
    lon_min=${spatial_rngs[$(( ( i - 1 ) * $nparms + 2 ))]}
    lon_max=${spatial_rngs[$(( ( i - 1 ) * $nparms + 3 ))]}
    lat_nbox=${spatial_rngs[$(( ( i - 1 ) * $nparms + 4 ))]}
    lon_nbox=${spatial_rngs[$(( ( i - 1 ) * $nparms + 5 ))]}


    time_str=$( printf "%04d-%04d" $beg_year $end_year )
    spatial_str=$( printf "%s-%s-n%d_%s-%s-n%d" $( pretty_lat $lat_min ) $( pretty_lat $lat_max ) $lat_nbox $( pretty_lon $lon_min ) $( pretty_lon $lon_max ) $lon_nbox )


    output_dir=$output_root/${time_str}_${spatial_str}

    echo "time_str    : $time_str"    
    echo "spatial_str : $spatial_str"
    echo "extra_suffix: $extra_suffix"
    echo "output_dir  : $output_dir"

    mkdir -p $output_dir

    python3 compute_regrid_budget/construct_timeseries_by_boxes_parallel.py \
        --beg-year=$beg_year        \
        --end-year=$end_year        \
        --output-dir $output_dir    \
        --lat-rng $lat_min $lat_max \
        --lon-rng $lon_min $lon_max \
        --lat-nbox $lat_nbox        \
        --lon-nbox $lon_nbox        \
        --mask-ERA $mask_ERA5       \
        --mask-ECCO $mask_ECCO      \
        --regrid-file-ERA5 $regrid_file_ERA5 \
        --regrid-file-ECCO $regrid_file_ECCO \
        --ignore-empty-box          \
        --ERA-type ERA5             \
        --nproc $nproc 

done
