#!/bin/bash

source 000_setup.sh
source tools/pretty_latlon.sh

nproc=41
moving_avg_days=31

input_root=$gendata_dir/regrid
output_root=$gendata_dir/anomalies

params=(
    1993 2016 1993-2016_31S-31N-n31_100E-100W-n80
)

nparms=3

for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    beg_year=${params[$(( ( i - 1 ) * $nparms + 0 ))]}
    end_year=${params[$(( ( i - 1 ) * $nparms + 1 ))]}
    dataset=${params[$(( ( i - 1 ) * $nparms + 2 ))]}

    input_dir=$input_root/$dataset
    output_dir=$output_root/$dataset/${beg_year}-${end_year}

    echo "beg_year   : $beg_year"
    echo "end_year   : $end_year"
    echo "datasetr   : $beg_year"
    echo "input_dir  : $input_dir"
    echo "output_dir : $output_dir"

    mkdir -p $output_dir

    time python3 compute_regrid_budget/compute_anomalies.py \
        --nproc $nproc                     \
        --year-rng $beg_year $end_year     \
        --input-dir  $input_dir            \
        --output-dir $output_dir           \
        --moving-avg-days $moving_avg_days \
        --vertical-layers-needed 20        \
        --nproc $nproc 

done
