#!/bin/bash
#SBATCH --job-name=CPOX
#SBATCH --error=error.log
#SBATCH --output=output.log
#SBATCH -n1
#SBATCH --partition=west,short
#SBATCH --exclude=c5003
#SBATCH --mem=25Gb
#SBATCH --time=24:00:00

source activate rmg
python-jl  $RMGpy/rmg.py input.py
