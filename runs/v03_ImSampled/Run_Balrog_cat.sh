#!/bin/bash
#SBATCH --job-name balrog_concat
##SBATCH --partition=broadwl
#SBATCH --partition=chihway
#SBATCH --account=pi-chihway
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --mail-user=dhayaa@uchicago.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/runs/v03_ImSampled/log_MakeBalrog

if [ "$USER" == "dhayaa" ]
then
    source /home/dhayaa/.bashrc
    cd /home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/
    conda activate shearDM
    source /home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/bash_profile.sh
fi

export PATH=/home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/:${PATH}
export PYTHONPATH=/home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/:${PYTHONPATH}
SCRIPT_DIR=/home/dhayaa/Desktop/DECADE/Balrog_of_the_stars/runs/v03_ImSampled/

python -u ${SCRIPT_DIR}/Make_balrog_cat.py
