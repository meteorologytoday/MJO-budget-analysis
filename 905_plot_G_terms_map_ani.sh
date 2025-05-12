#!/bin/bash

source 000_setup.sh

input_root=$gendata_dir/1993-2016_31S-31N-n31_100E-100W-n80/1993-2016/bandpass
output_root=$fig_dir/Gterms_map
    
mkdir -p $output_root

params=(
    CASE1 2015-10-01  2016-04-30 -30 30 30 270
)

nparms=7


for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    casename=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    time_beg=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    time_end=${params[$(( ( i - 1 ) * $nparms + 2 ))]}
    lat_min=${params[$(( ( i - 1 ) * $nparms + 3 ))]}
    lat_max=${params[$(( ( i - 1 ) * $nparms + 4 ))]}
    lon_min=${params[$(( ( i - 1 ) * $nparms + 5 ))]}
    lon_max=${params[$(( ( i - 1 ) * $nparms + 6 ))]}
    

   
    echo "casename : $casename"
    echo "time_beg : $time_beg"
    echo "time_end : $time_end"
    echo "lat_min  : $lat_min"
    echo "lat_max  : $lat_max"
    echo "lon_min  : $lon_min"
    echo "lon_max  : $lon_max"
    echo "input_root  : $input_root"

    output_dir=$output_root/$casename
    mkdir -p $output_dir
 
    IFS=" "
    dts=( $( python3 tools/print_date_list.py --date-rng $time_beg $time_end ) )

    echo "dts=${dts[@]}"

    for dt in "${dts[@]}" ; do

        output_file=$output_dir/Gterms_map-${dt}.png


        if [ -f "$output_file" ] ; then
            echo "Output file $output_file already exists. Skip."

        else 
            echo "Making output_file : $output_file"
            python3 plotting/plot_G_terms_map.py    \
                --input-dir  $input_root           \
                --output     $output_file          \
                --time-rng   $dt $dt   \
                --lat-rng    $lat_min $lat_max     \
                --lon-rng    $lon_min $lon_max     \
                --ncol       2 \
                --varnames   dMLTdt      MLG_frc    \
                             MLG_nonfrc  MLG_frc_sw \
                             MLG_adv     MLG_frc_lw \
                             MLG_vmixall MLG_frc_sh \
                             BLANK       MLG_frc_lh \
                --no-display --add-thumbnail-title
        fi
    done

done
