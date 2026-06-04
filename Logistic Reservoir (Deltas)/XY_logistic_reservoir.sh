#!/bin/bash
#PBS -N XY_logistic_reservoir
#PBS -o outxy.log
#PBS -e errxy.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 logistic_reservoir_xy.py