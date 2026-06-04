#!/bin/bash
#PBS -N YX_logistic_reservoir
#PBS -o outyz.log
#PBS -e erryz.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 logistic_reservoir_yx.py