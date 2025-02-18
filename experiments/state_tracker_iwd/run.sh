#!/bin/bash

if [ "$1" == "train" ]; then
    train_or_test="train"
    env_variant="\"\""
    echo "Training environment"
elif [ "$1" == "test_fixed" ]; then
    train_or_test="test"
    env_variant="fixed"
    echo "Testing fixed circle environment"
elif [ "$1" == "test_fixed_cw" ]; then
    train_or_test="test"
    env_variant="fixed_cw"
    echo "Testing fixed circle (clockwise) environment"
elif [ "$1" == "test_eight" ]; then
    train_or_test="test"
    env_variant="eight"
    echo "Testing eight drift environment"
elif [ "$1" == "test_cw" ]; then
    train_or_test="test"
    env_variant="cw"
    echo "Testing fixed circle cw environment"
else
    echo "Invalid argument. Expected 'train', 'test_fixed', 'test_fixed_cw' or 'test_eight'."
    exit 1
fi

python ../../run.py $train_or_test state_tracker_iwd --mlp-size-last 16 --horizon 20 --num-parallel 300000 --lr-schedule adaptive --epochs 1000 --env-variant $env_variant --disturbed --randomize-tyre small --exp-name default --quiet --car-preset xcar
