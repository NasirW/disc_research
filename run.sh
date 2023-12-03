#!/bin/zsh

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                             #
#                     Autorun Script File For DISC Script                     #
#                                                                             #
#                             ./run.sh (...)                                  #
#                                                                             #
#                           for usage: ./run.sh                               #
#                                                                             #
#                   (Must Use `chmod +x run.sh` once first)                   #
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
    echo_reset "${RED}Usage: ./run.sh (batch | lbatch | clean)"
    echo_reset "${YELLOW}batch: ${BLUE}runs batch job on cluster"
    echo_reset "${YELLOW}local: ${BLUE}runs batch job locally"
    echo_reset "${YELLOW}clean: ${BLUE}cleans up file space"
    exit 1
}

echo_and_run() {
    echo_reset "${ITALICS}${CYAN}$1\n" 
    eval $1
}




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                               Argument Handling                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

clear

if [[ $# -eq 0 ]]; then
    usage_error
fi


module=$1
shift
args=${*}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                 Paths/Files                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

PYTHON=python3.11

HANDLE_DATA=encoders/handle_data.py
ENCODE_DATA=encoders/encode_data.py
DATA_MODE=padchest
DATA_DIR=data/padchest
MAIN=src/main.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Input Mapping                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

case ${module} in
    clean)
        echo_reset "${PURPLE}Cleaning Up File Space..."
        echo_and_run "rm *.out"
        echo_and_run "rm *.err"
        ;;
    batch)
        echo_reset "${PURPLE}Running Batch Job...\n"
        ./run.sh clean
        sbatch gpu_run.sh
        ;;
    local)
        echo_reset "${PURPLE}Running Batch Job Locally...\n"
        ./gpu_run.sh
        ;;
    *)
        usage_error
        ;;
esac