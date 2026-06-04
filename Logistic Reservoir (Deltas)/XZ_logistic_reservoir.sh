#!/bin/bash
#PBS -N XZ_logistic_reservoir
#PBS -o outxz.log
#PBS -e errxz.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 logistic_reservoir_xz.py