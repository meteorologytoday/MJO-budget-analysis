#!/bin/bash

source 000_setup.sh

dataset=1993-2016_31S-31N-n31_100E-100W-n80

python3 compute_regrid_budget/compute_extra_variables.py \
    --input-dir $gendata_dir/$dataset/1993-2016/anom

