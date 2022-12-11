#!/bin/bash

OUTNAME=sound_recordings_fixed_ts_1.bag
source .venv/bin/activate
rosbag record --all -O $OUTNAME
#python read_serial.py

