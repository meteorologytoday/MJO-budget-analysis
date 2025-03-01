#!/bin/bash


output_dir=figures
mkdir -p $output_dir

beg_year=1993
end_year=2017

yrng_str="${beg_year}-${end_year}"
data_dir=output_ECCO/${yrng_str}_10N-60N-n25_120E-120W-n60




python3 plot_AR_histogram.py \
    --input-dir $data_dir \
    --climanom-input-dir $data_dir/climanom_${yrng_str} \
    --beg-year $beg_year \
    --end-year $end_year \
    --output $output_dir/histogram.png &


echo "Fig S1"
#python3 plot_AR_basic_diagnostics.py --input $data_dir/AR_simple_statistics_${yrng_str}.nc --output $output_dir/AR_simple_stat.png &

if [ ] ; then
echo "Fig S2"
python3 plot_AR_freq_EOF.py \
    --input $data_dir/EOF.nc \
    --output-EOF        $output_dir/AR_EOF.png \
    --output-timeseries $output_dir/AR_EOF_timeseries.png 
fi

echo "Fig 1"

python3 plot_G_terms_map.py \
    --input-dir $data_dir/climanom_${yrng_str} \
    --output $output_dir/G_terms_atmocn.png \
    --watermonths 7 \
    --no-display &

python3 plot_G_terms_map.py \
    --input-dir $data_dir/climanom_${yrng_str} \
    --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLG_hdiff MLD \
    --output $output_dir/G_terms_ocn.png \
    --watermonths 7 \
    --no-display &

python3 plot_G_terms_map.py \
    --input-dir $data_dir/climanom_${yrng_str} \
    --varnames MLG_frc MLG_frc_sw MLG_frc_lw MLG_frc_sh MLG_frc_lh MLG_frc_fwf \
    --output $output_dir/G_terms_atm.png \
    --watermonths 7 \
    --no-display &


#python3 plot_G_terms_map.py --input-dir $data_dir/climanom_IWV-off_${yrng_str} --output $output_dir/G_terms_atmocn_IWV-off.png --no-display &


#python3 plot_G_terms_map.py --input-dir $data_dir/climanom_IWV-off_${yrng_str} --varnames MLG_nonfrc MLG_adv MLG_vdiff MLG_ent MLD MXLDEPTH --output $output_dir/G_terms_ocn_IWV-off.png --no-display &


echo "Fig 2"
#python3 plot_AR_freq_with_std.py \
#    --input $data_dir/AR_interannual_statistics_${yrng_str}.nc \
#    --output $output_dir/AR_freq_std.png &

if [ ] ; then
python3 plot_AR_freq.py \
    --input-dir $data_dir \
    --output $output_dir/AR_freq1.png \
    --beg-year 1993 \
    --end-year 2017  \
    --freq-max 0.50 \
    --ndays 1       \
    --threshold-days 1 &


python3 plot_AR_freq.py \
    --input-dir $data_dir \
    --output $output_dir/AR_freq2.png \
    --beg-year 1993 \
    --end-year 2017  \
    --freq-max 0.05 \
    --ndays 7 \
    --threshold-days 7 &

python3 plot_AR_freq.py \
    --input-dir $data_dir \
    --output $output_dir/AR_freq3.png \
    --beg-year 1993 \
    --end-year 2017 \
    --freq-max 0.05  \
    --ndays 10 \
    --threshold-days 8 \
    --markers &


fi

echo "Fig 3"
#python3 plot_G_terms_map.py --input-dir $data_dir/climanom_${yrng_str} --output $output_dir/AR_forcing_partition1.png --no-display

echo "Fig 4"
#python3 plot_G_terms_map.py --input-dir $data_dir --output $output_dir/AR_forcing_partition1.png --no-display

echo "Fig 5"

. pretty_latlon.sh
spatial_rngs=(
    30 210
)
if [ ] ; then
nparms=2
for i in $( seq 1 $(( ${#spatial_rngs[@]} / $nparms )) ); do

    lat=${spatial_rngs[$(( ( i - 1 ) * $nparms + 0 ))]}
    lon=${spatial_rngs[$(( ( i - 1 ) * $nparms + 1 ))]}
    
    for bkdn in atmocn atm ocn ; do

        output_filename="$output_dir/Gterms_$( pretty_lat $lat )_$( pretty_lon $lon )_${bkdn}.png"
        python3 plot_G_terms_point.py \
            --input-dir $data_dir/climanom_${yrng_str}     \
            --lat $lat                \
            --lon $lon                \
            --breakdown $bkdn         \
            --title-style latlon      \
            --output $output_filename &
        
        
    done
done

fi
wait
