#!/bin/zsh

#SBATCH --partition preempt
#SBATCH --job-name=run
#SBATCH --output=%j_out.ans
#SBATCH --error=%j_err.ans
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

echo_run_halt() {
    echo_and_run "${*}"

    exit_code=$?

    if [ ${exit_code} -ne 0 ]; then
        echo_reset "\n${RED}Failiure with Exit Code {${exit_code}} on:\n${ITALICS}${1}"
        exit ${exit_code}
    fi
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                 Paths/Vars                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# PYTHON=/cluster/tufts/hugheslab/nwynru01/condaenv/multiview/bin/python
PYTHON=python3

PREP=Data_prep.py
SUSP=Data_prep_suspicious.py
TRAIN=TrainDenseNET.py
EVAL=EvaluateCNN.py
TEST=test.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Input Mapping                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

echo_run_halt "${PYTHON} ${TEST}"