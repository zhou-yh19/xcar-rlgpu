#!/bin/sh
python ../../run.py train continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar

python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode hybrid
python plot.py --filename hybrid_ref_exp
python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode circle
python plot.py --filename circle_ref_exp
python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode eight
python plot.py --filename eight_ref_exp
python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode three
python plot.py --filename three_ref_exp
python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode olympic
python plot.py --filename olympic_ref_exp
python ../../run.py test continuous_drift_iwd --mlp-size-last 16 --horizon 20 --num-parallel 20000 --lr-schedule adaptive --epochs 1000 $env_variant --disturbed --randomize-tyre small --exp-name continuous_drift_iwd_default --car-preset xcar --ref-mode variable_curvature
python plot.py --filename variable_curvature_ref_exp
