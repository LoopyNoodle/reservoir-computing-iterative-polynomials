#!/bin/bash
#PBS -N error_ar_et_al
#PBS -o aetal_o.log
#PBS -e aetal_e.log
#PBS -l ncpus=5
#PBS -q cpu

module load compiler/anaconda3
python3 logistic_res_task2.2_arunetal.py