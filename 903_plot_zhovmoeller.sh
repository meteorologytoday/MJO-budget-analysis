#!/bin/bash

source 000_setup.sh
source tools/pretty_latlon.sh

input_root=$gendata_dir/1993-2016_31S-31N-n31_100E-100W-n80/1993-2016
output_root=$fig_dir/zhovmoeller
    
mkdir -p $output_root

params=(
     2010-10-01 2011-09-30 -1 1 148 150
     2011-10-01 2012-09-30 -1 1 148 150
     2012-10-01 2013-09-30 -1 1 148 150
     2013-10-01 2014-09-30 -1 1 148 150
     2014-10-01 2015-09-30 -1 1 148 150
     2015-10-01 2016-09-30 -1 1 148 150

     2010-10-01 2011-09-30 -1 1 178 180
     2011-10-01 2012-09-30 -1 1 178 180
     2012-10-01 2013-09-30 -1 1 178 180
     2013-10-01 2014-09-30 -1 1 178 180
     2014-10-01 2015-09-30 -1 1 178 180
     2015-10-01 2016-09-30 -1 1 178 180

)

if [ ] ; then
params=(
     2010-10-01 2011-09-30 -1 1 178 180
     2011-10-01 2012-09-30 -1 1 178 180
     2012-10-01 2013-09-30 -1 1 178 180
     2013-10-01 2014-09-30 -1 1 178 180
     2014-10-01 2015-09-30 -1 1 178 180
     2015-10-01 2016-09-30 -1 1 178 180
)


fi

params=(
     2015-10-01 2016-09-30 -5 5 170 180
     2015-10-01 2016-09-30 -5 5 140 150
)


nparms=6

for filter in bandpass-mavg bandpass-lanczos bandpass-hat anom ; do
for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    time_beg=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    time_end=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    lat_min=${params[$(( ( i - 1 ) * $nparms + 2 ))]}
    lat_max=${params[$(( ( i - 1 ) * $nparms + 3 ))]}
    lon_min=${params[$(( ( i - 1 ) * $nparms + 4 ))]}
    lon_max=${params[$(( ( i - 1 ) * $nparms + 5 ))]}
    
    spatial_str=$( printf "%s-%s_%s-%s" $( pretty_lat $lat_min ) $( pretty_lat $lat_max ) $( pretty_lon $lon_min ) $( pretty_lon $lon_max ) )
    
    output_dir=$output_root/$filter
    output_file=$output_dir/zhov_${filter}_${time_beg}_to_${time_end}_${spatial_str}.png

    echo "time_beg      : $time_beg"
    echo "time_end      : $time_end"
    echo "lat_min       : $lat_min"
    echo "lat_max       : $lat_max"
    echo "lon_min       : $lon_min"
    echo "lon_max       : $lon_max"

    echo "input_root : $input_root"
    echo "output_file : $output_file"

    mkdir -p $output_dir

    if [ -f "$output_file" ]; then

        echo "Output file $output_file exists. Skip"

    else
        echo "Making file $output_file "

#            --varname    $filter::ttr $filter::EXFpreci $filter::THETA $filter::SALT $filter::Ue       \
        python3 plotting/plot_hovmoeller_in_z.py    \
            --varname    $filter::ttr $filter::EXFpreci $filter::MLT       \
            --input-root $input_root           \
            --output     $output_file          \
            --time-rng   $time_beg $time_end   \
            --lat-rng    $lat_min $lat_max     \
            --lon-rng    $lon_min $lon_max     \
            --depth 200 \
            --no-display \
            --show-title --show-numbering

    fi
done
done
