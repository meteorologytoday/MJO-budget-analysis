#!/bin/bash

source 000_setup.sh
source 999_trapkill.sh

source tools/pretty_latlon.sh


nproc=10

dataset=1993-2016_31S-31N-n31_100E-100W-n80

input_root=$gendata_dir/anomalies/$dataset/1993-2016
output_root=$gendata_dir/bandpass/$dataset/1993-2016

mkdir -p $output_root

params=(

    SALT
    THETA
    Ue
    Vn

    EXFpreci

    MLT

    MLG_frc
    MLG_nonfrc 
    dMLTdt
   
    MLG_frc_sw
    MLG_frc_lw
    MLG_frc_sh
    MLG_frc_lh
    MLG_frc_dilu
    MLG_adv
    MLG_vmix
    MLG_hdiff
    MLG_vdiff
    MLG_ent_wen

    ttr
)

params=(
    MLT
    THETA
    SALT
    Ue
    Vn
)

nparms=1
for i in $( seq 1 $(( ${#params[@]} / $nparms )) ); do

    varname=${params[$(( ( i - 1 ) * $nparms + 0 ))]}

    echo "varname : $varname"

    time python3 compute_bandpass/genMJOsignal.py \
        --varname    $varname                \
        --input-dir  $input_root              \
        --output-dir $output_root          &

    cnt=$(( $cnt + 1))
    
    if (( $cnt >= $nproc )) ; then
        echo "Max cpu usage reached: $nproc"
        wait
        cnt=0
    fi 

done

wait

echo "All done"
