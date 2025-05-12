#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

source tools/pretty_latlon.sh

nproc=10
beg_time="1998-01-01"
end_time="2017-12-31"

beg_time="2015-12-01"
end_time="2016-01-31"

beg_time="2011-11-21"
end_time="2011-12-10"


output_root=$gendata_dir/LPO

mkdir -p $output_root

params=(
    ERA5
)

nparms=1

for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    dataset=${params[$(( ( i - 1 ) * $nparms + 0 ))]}

    echo "dataset : $dataset"

    time python3 MJO_detection/gen_detect_maps.py \
        --output-root $output_root \
        --nproc $nproc     \
        --dataset $dataset \
        --beg-time $beg_time \
        --end-time $end_time 


done

wait

echo "All done"
