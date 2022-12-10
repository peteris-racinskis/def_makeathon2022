#!/bin/bash

OUTNAME=sound_recordings_unified-freq-timed-7_hf.bag
source .venv/bin/activate
rosbag record --all -O $OUTNAME
#python read_serial.py

