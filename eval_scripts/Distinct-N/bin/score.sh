#!/usr/bin/env bash

HYPO=../testdata/bigram.txt
DIR=/home/cgsdfc/Result/Test

#python bin/distinct_metric.py  $HYPO -n 3

HYPO=./testdata/bigram.txt
python bin/distinct_metric.py  $HYPO -n 3