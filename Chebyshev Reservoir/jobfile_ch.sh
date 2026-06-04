#!/bin/bash
#PBS -N chebyshev_delta
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 chebyshev_delta.py