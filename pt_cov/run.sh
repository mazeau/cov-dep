#!/bin/bash
#SBATCH --job-name=PTc_cpox
#SBATCH --error=error.log
#SBATCH --output=output.log
#SBATCH -n1
#SBATCH --partition=west,short
#SBATCH --exclude=c5003
#SBATCH --mem=10Gb
#SBATCH --time=24:00:00

source activate rmg_env
python  $RMGpy/rmg.py input.py
