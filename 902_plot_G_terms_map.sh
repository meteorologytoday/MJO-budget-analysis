#!/bin/bash

source 00_setup.sh

input_root=$gendata_dir/bandpass/1993-2017_31S-31N-n31_100E-100W-n80
output_root=$fig_dir/Gterms_map
    
mkdir -p $output_root

params=(
    2016-01-16  2016-01-31 -30 30 30 270
    2015-12-16  2015-12-31 -30 30 30 270
    2011-11-01  2011-11-15 -30 30 30 270
    2012-03-01  2012-03-15 -30 30 30 270
)

nparms=6


for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    time_beg=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    time_end=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    lat_min=${params[$(( ( i - 1 ) * $nparms + 2 ))]}
    lat_max=${params[$(( ( i - 1 ) * $nparms + 3 ))]}
    lon_min=${params[$(( ( i - 1 ) * $nparms + 4 ))]}
    lon_max=${params[$(( ( i - 1 ) * $nparms + 5 ))]}
    
    output_file=$output_root/Gterms_map-${time_beg}_to_${time_end}.png

    
    echo "time_beg : $time_beg"
    echo "time_end : $time_end"
    echo "lat_min  : $lat_min"
    echo "lat_max  : $lat_max"
    echo "lon_min  : $lon_min"
    echo "lon_max  : $lon_max"

    echo "input_root  : $input_root"
    echo "output_file : $output_file"

    python3 plotting/plot_G_terms_map.py    \
        --input-dir  $input_root           \
        --output     $output_file          \
        --time-rng   $time_beg $time_end   \
        --lat-rng    $lat_min $lat_max     \
        --lon-rng    $lon_min $lon_max     \
        --ncol       2 \
        --varnames   dMLTdt      MLG_frc    \
                     MLG_nonfrc  MLG_frc_sw \
                     MLG_adv     MLG_frc_lw \
                     MLG_vmix    MLG_frc_sh \
                     MLG_ent_wen MLG_frc_lh \
        --no-display
#                     MLG_hdiff   MLG_frc_dilu \

done
