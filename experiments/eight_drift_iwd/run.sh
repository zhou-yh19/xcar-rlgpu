#!/bin/sh
python ../../run.py train eight_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 300000 --lr-schedule adaptive --epochs 500 $env_variant --disturbed --randomize-tyre small --exp-name default --car-preset xcar
python ../../run.py test eight_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 300000 --lr-schedule adaptive --epochs 500 $env_variant --disturbed --randomize-tyre small --exp-name default --car-preset xcar
python plot.py --filename eight_exp