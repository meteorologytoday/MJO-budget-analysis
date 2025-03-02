#!/bin/bash

source 00_setup.sh

input_root=$gendata_dir/bandpass
output_root=$fig_dir/hovmoeller
    
mkdir -p $output_root

params=(
    ttr 2011-10-01 2012-04-01 -5 5 30 230
)

nparms=7

for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    varname=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    time_beg=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    time_end=${params[$(( ( i - 1 ) * $nparms + 2 ))]}
    lat_min=${params[$(( ( i - 1 ) * $nparms + 3 ))]}
    lat_max=${params[$(( ( i - 1 ) * $nparms + 4 ))]}
    lon_min=${params[$(( ( i - 1 ) * $nparms + 5 ))]}
    lon_max=${params[$(( ( i - 1 ) * $nparms + 6 ))]}
    
    input_file=$input_root/anom_${varname}.nc
    output_file=$output_root/hov-anom_${varname}.png

    
    echo "varname  : $varname"
    echo "time_beg : $time_beg"
    echo "time_end : $time_end"
    echo "lat_min  : $lat_min"
    echo "lat_max  : $lat_max"
    echo "lon_min  : $lon_min"
    echo "lon_max  : $lon_max"

    echo "input_file : $input_file"
    echo "output_file : $output_file"

    python3 plotting/plot_hovemoller.py    \
        --varname    $varname              \
        --input-file $input_file           \
        --output     $output_file          \
        --time-rng   $time_beg $time_end   \
        --lat-rng    $lat_min $lat_max     \
        --lon-rng    $lon_min $lon_max     \
        --reverse-time

done
