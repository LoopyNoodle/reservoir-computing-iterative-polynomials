#!/bin/bash
#PBS -N tent_delta
#PBS -o out.log
#PBS -e err.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 tent_delta.py