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
    echo_reset "${RED}Usage: ./run.sh (clean | batch | local | data | biop | susp | meta)"
    echo_reset "${YELLOW}clean: ${BLUE}cleans up file space"
    echo_reset "${YELLOW}batch: ${BLUE}runs batch job on cluster"
    echo_reset "${YELLOW}local: ${BLUE}runs batch job locally"
    echo_reset "${YELLOW}data:  ${BLUE}runs data prep module"
    echo_reset "${YELLOW}biop:  ${BLUE}runs data prep module for biopsy data"
    echo_reset "${YELLOW}susp:  ${BLUE}runs data prep module for suspicious data"
    echo_reset "${YELLOW}meta:  ${BLUE}runs data prep module for metastasis data"
    exit 1
}

echo_and_run() {
    echo_reset "${ITALICS}${CYAN}$1\n" 
    eval $1
}

clean() {
    echo_reset "${PURPLE}Cleaning Up File Space..."
    echo_and_run "rm *.ans"
    echo_and_run "rm *.png"
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


BIOP=chemo_res_biop
BIOP_RES=data/chemo_res_biop/Norm_Resistant
BIOP_SEN=data/chemo_res_biop/Norm_Sensitive

SUSP=chemo_res_susp
SUSP_RES=data/chemo_res_susp/Norm_Resistant
SUSP_SEN=data/chemo_res_susp/Norm_Sensitive

META=metastasis
META_MAL=data/metastasis/Classification/Metastasis
META_BEN=data/metastasis/Classification/Benign

DATA_PREP=Data_prep_class.py

dp_biop_args="--res_folder ${BIOP_RES} --sen_folder ${BIOP_SEN} --name ${BIOP}"
dp_susp_args="--res_folder ${SUSP_SEN} --sen_folder ${SUSP_RES} --name ${SUSP}"
dp_meta_args="--res_folder ${META_BEN} --sen_folder ${META_MAL} --name ${META}"



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                Input Mapping                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

case ${module} in
    clean)
        clean
        ;;
    batch)
        clean
        echo_reset "${PURPLE}Running Batch Job...\n"
        sbatch gpu_run.sh
        ;;
    local)
        echo_reset "${PURPLE}Running Batch Job Locally...\n"
        ./gpu_run.sh
        ;;
    data)
        echo_reset "${PURPLE}Running Data Prep Module...\n"
        echo_and_run "${PYTHON} ${DATA_PREP} ${args}"
        ;;
    biop)
        echo_reset "${PURPLE}Running Data Prep Module for Biopsy Data...\n"
        echo_and_run "${PYTHON} ${DATA_PREP} ${dp_biop_args} ${args}"
        ;;
    susp)
        echo_reset "${PURPLE}Running Data Prep Module for Suspended Data...\n"
        echo_and_run "${PYTHON} ${DATA_PREP} ${dp_susp_args} ${args}"
        ;;
    meta)
        echo_reset "${PURPLE}Running Data Prep Module for Metastasis Data...\n"
        echo_and_run "${PYTHON} ${DATA_PREP} ${dp_meta_args} ${args}"
        ;;
    *)
        usage_error
        ;;
esac