#!/bin/sh
python ../../run.py train fixed_circle_iwd --mlp-size-last 16 --horizon 20 --num-parallel 100000 --lr-schedule adaptive --epochs 500 $env_variant --disturbed --randomize-tyre small --exp-name default  --quiet --car-preset xcar
python ../../run.py test fixed_circle_iwd --mlp-size-last 16 --horizon 20 --num-parallel 100000 --lr-schedule adaptive --epochs 500 $env_variant --disturbed --randomize-tyre small --exp-name default  --quiet --car-preset xcar
python plot.py --filename circle_exp