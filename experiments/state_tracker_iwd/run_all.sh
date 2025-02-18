#!/bin/bash
./run.sh train
./run.sh test_fixed
python plot.py --filename fixed_exp
./run.sh test_fixed_cw
python plot.py --filename fixed_cw_exp
./run.sh test_eight
python plot.py --filename eight_exp