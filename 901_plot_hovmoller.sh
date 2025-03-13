#!/bin/bash

source 00_setup.sh

input_root=$gendata_dir
dataset=1993-2017_31S-31N-n31_100E-100W-n80
output_root=$fig_dir/hovmoeller
    
mkdir -p $output_root

params=(
#    ttr     bandpass   2015-10-01 2016-03-01 -5 5 30 270
#    dMLTdt  bandpass   2015-10-01 2016-03-01 -5 5 30 270
    EXFpreci ttl  2015-10-01 2016-03-01 -5 5 30 270
    EXFpreci anomalies  2015-10-01 2016-03-01 -5 5 30 270
    EXFpreci bandpass  2015-10-01 2016-03-01 -5 5 30 270
    MLT     bandpass   2015-10-01 2016-03-01 -5 5 30 270

    ttr     anomalies  2015-10-01 2016-03-01 -5 5 30 270
    dMLTdt  anomalies  2015-10-01 2016-03-01 -5 5 30 270
#    MLT     anomalies  2015-05-01 2016-03-01 -5 5 30 270
)

nparms=8

for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    varname=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    filtered_type=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    time_beg=${params[$(( ( i - 1 ) * $nparms + 2 ))]}
    time_end=${params[$(( ( i - 1 ) * $nparms + 3 ))]}
    lat_min=${params[$(( ( i - 1 ) * $nparms + 4 ))]}
    lat_max=${params[$(( ( i - 1 ) * $nparms + 5 ))]}
    lon_min=${params[$(( ( i - 1 ) * $nparms + 6 ))]}
    lon_max=${params[$(( ( i - 1 ) * $nparms + 7 ))]}
    
    output_dir=$output_root/$filtered_type


    bad_naming=$filtered_type

    if [ "$filtered_type" = "ttl" ] ; then
        bad_naming="anomalies"
    fi

    input_file=$input_root/$bad_naming/$dataset/${filtered_type}_${varname}.nc
    output_file=$output_dir/hov-${filtered_type}_${varname}_${time_beg}_to_${time_end}.png

    echo "varname       : $varname"
    echo "filtered_type : $filtered_type"
    echo "time_beg      : $time_beg"
    echo "time_end      : $time_end"
    echo "lat_min       : $lat_min"
    echo "lat_max       : $lat_max"
    echo "lon_min       : $lon_min"
    echo "lon_max       : $lon_max"

    echo "input_file : $input_file"
    echo "output_file : $output_file"

    mkdir -p $output_dir

    python3 plotting/plot_hovemoller.py    \
        --varname    $varname              \
        --filtered-type $filtered_type     \
        --input-file $input_file           \
        --output     $output_file          \
        --time-rng   $time_beg $time_end   \
        --lat-rng    $lat_min $lat_max     \
        --lon-rng    $lon_min $lon_max     \
        --aspect-ratio 1 \
        --reverse-time --no-display \
        --show-title --show-numbering --numbering $i

done
