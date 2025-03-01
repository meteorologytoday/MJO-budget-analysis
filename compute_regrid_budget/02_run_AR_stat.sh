#!/bin/bash

ncpu=51
beg_year=1993
end_year=2017

input_dir=output_ECCO/1993-2017_10N-60N-n25_100E-100W-n80
yrng_str="${beg_year}-${end_year}"
annual_cnt_threshold=10

for annual_cnt_threshold in 5; do
    for AR_algo in "HMGFSC24_threshold-1998-2017" ; do

        _input_dir="${input_dir}"
        python3 mk_AR_interannual_stat.py --input $_input_dir --beg-year $beg_year --end-year $end_year --AR-algo $AR_algo 
        python3 mk_AR_EOF.py --input $_input_dir/AR_interannual_statistics_${AR_algo}_${yrng_str}.nc --output $_input_dir/EOF_${AR_algo}.nc
        python3 mk_AR_stat_stderr.py --input-dir $_input_dir --beg-year $beg_year --end-year $end_year --ncpu $ncpu --AR-algo $AR_algo --annual-cnt-threshold $annual_cnt_threshold
    done
done





