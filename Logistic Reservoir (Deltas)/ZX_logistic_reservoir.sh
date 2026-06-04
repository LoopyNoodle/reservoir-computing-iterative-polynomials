#!/bin/bash
#PBS -N ZX_logistic_reservoir
#PBS -o outzx.log
#PBS -e errzx.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 logistic_reservoir_zx.py