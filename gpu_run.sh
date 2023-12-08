#!/bin/bash

#SBATCH --partition preempt
#SBATCH --job-name=run
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=30:00:00
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=100000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nasir.wynruit@tufts.edu


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#                             GPU QUICK RUN SCRIPT                            #
#                                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                     COLOR                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

END='\e[0m'

ITALICS='\e[3m'
RED='\e[1;31m'
GREEN='\e[1;32m'
YELLOW='\e[1;33m'
BLUE='\e[1;34m'
PURPLE='\e[1;35m'
CYAN='\e[1;36m'
MAGENTA='\e[1;95m'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                  FUNCTIONS                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

echo_reset() {
    echo -e "$*${END}"
}

usage_error() {
    echo_reset "${RED}Usage: ./gpu_run.sh"
    exit 1
}

echo_and_run() {
    echo_reset "${ITALICS}${CYAN}$1\n" 
    eval $1
}



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                 Paths/Vars                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# todo: check other python versions
conda init bash
conda activate /cluster/tufts/hpc/tools/anaconda/202105/envs/pytorch_cuda11.7
PYTHON=/cluster/tufts/hugheslab/nwynru01/condaenv/multiview/bin/python
# PYTHON=/cluster/tufts/hugheslab/nwynru01/condaenv/multiview/bin/python



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Input Mapping                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

python cuda_check.py