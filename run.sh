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
    echo_reset "${RED}Usage: ./run.sh (clean | batch | local | data <args> | biop [<args>] | susp [<args>] | meta [<args>] | model <args>)"
    echo_reset "${YELLOW}clean: ${BLUE}cleans up file space"
    echo_reset "${YELLOW}batch: ${BLUE}runs batch job on cluster"
    echo_reset "${YELLOW}local: ${BLUE}runs batch job locally"
    echo_reset "${YELLOW}data:  ${BLUE}runs data prep module with args"
    echo_reset "${YELLOW}biop:  ${BLUE}runs data prep module for biopsy data with optional args"
    echo_reset "${YELLOW}susp:  ${BLUE}runs data prep module for suspicious data with optional args"
    echo_reset "${YELLOW}meta:  ${BLUE}runs data prep module for metastasis data with optional args"
    echo_reset "${YELLOW}model: ${BLUE}runs model module with args"
    exit 1
}

echo_run() {
    echo_reset "${ITALICS}${CYAN}$1\n" 
    eval $1
}

echo_run_halt() {
    echo_run "$1"

    exit_code=$?

    if [ ${exit_code} -ne 0 ]; then
        echo_reset "\n${RED}Exited with {${exit_code}} on:\n${ITALICS}${1}"
        exit ${exit_code}
    fi
}

clean() {
    echo_reset "${PURPLE}Cleaning Up File Space..."
    echo_run "rm *.ans"
    echo_run "rm *.png"
}

python() {
    echo_run "${PYTHON} $*"
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

IMG_LOAD=ImageLoader.py
MODEL=Model.py

BIOP=chemo_res_biop
BIOP_RES=data/${BIOP}/Norm_Resistant
BIOP_SEN=data/${BIOP}/Norm_Sensitive
BIOP_NPY=results/${BIOP}/npy

SUSP=chemo_res_susp
SUSP_RES=data/${SUSP}/Norm_Resistant
SUSP_SEN=data/${SUSP}/Norm_Sensitive
SUSP_NPY=results/${SUSP}/npy

META=metastasis
META_MAL=data/${META}/Classification/Metastasis
META_BEN=data/${META}/Classification/Benign
META_NPY=results/${META}/npy


il_biop_args="--res_folder ${BIOP_RES} --sen_folder ${BIOP_SEN} --name ${BIOP}"
il_susp_args="--res_folder ${SUSP_SEN} --sen_folder ${SUSP_RES} --name ${SUSP}"
il_meta_args="--res_folder ${META_BEN} --sen_folder ${META_MAL} --name ${META}"

model_biop_args="--data_dir ${BIOP_NPY}"
model_susp_args="--data_dir ${SUSP_NPY}"
model_meta_args="--data_dir ${META_NPY}"


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
        python "${IMG_LOAD} ${args}"
        ;;
    biop)
        echo_reset "${PURPLE}Running Data Prep Module for Biopsy Data...\n"
        python "${IMG_LOAD} ${il_biop_args} ${args}"
        ;;
    susp)
        echo_reset "${PURPLE}Running Data Prep Module for Suspicious Data...\n"
        python "${IMG_LOAD} ${il_susp_args} ${args}"
        ;;
    meta)
        echo_reset "${PURPLE}Running Data Prep Module for Metastasis Data...\n"
        python "${IMG_LOAD} ${il_meta_args} ${args}"
        ;;
    model)
        echo_reset "${PURPLE}Running Model Module...\n"
        python "${MODEL} ${args}"
        ;;
    modelb)
        echo_reset "${PURPLE}Running Model Module for Biopsy Data...\n"
        python "${MODEL} ${model_biop_args} ${args}"
        ;;
    models)
        echo_reset "${PURPLE}Running Model Module for Suspicious Data...\n"
        python "${MODEL} ${model_susp_args} ${args}"
        ;;
    modelm)
        echo_reset "${PURPLE}Running Model Module for Metastasis Data...\n"
        python "${MODEL} ${model_meta_args} ${args}"
        ;;
    *)
        usage_error
        ;;
esac