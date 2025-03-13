#!/bin/bash

source 00_setup.sh

dataset=1993-2017_31S-31N-n31_100E-100W-n80

python3 compute_regrid_budget/compute_extra_variables.py \
    --input-dir $gendata_dir/anomalies/$dataset

