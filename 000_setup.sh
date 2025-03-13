#!/bin/bash
py=python3
sh=bash

data_dir=./dataset
fig_dir=figures

gendata_dir=./gendata

export PYTHONPATH=`pwd`/tools:$PYTHONPATH

mkdir -p $fig_dir
