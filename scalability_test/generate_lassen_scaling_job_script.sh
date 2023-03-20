#!/bin/bash

NONOPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nodes)
            NUM_NODES="$2"
            shift
            shift
            ;;
        -e|--end)
            NUM_PROCS="$2"
            shift 
            shift
            ;;
        -s|--start)
            NUM_PROCS_1="$2"
            shift 
            shift
            ;;
        -o|--output)
            OUTPUT_PATH="$2"
            shift 
            shift
            ;;
        -q|--queue)
            QUEUE_NAME="$2"
            shift 
            shift
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 
            shift
            ;;
        -h|--help)
            printf "Generates batch script for scaling tests on Lassen.\n\nUsage:\n"
            printf "./generate_lassen_scaling_job_script.sh -n <nnodes> -s <nproc_start> -e <nproc_end> -o <batch script name> -q <queue name> -t <walltime limit>\n\n"
            printf "Default: scaling_test_lassen.bsub.sh = single node scaling job to run on 1-4 GPUs on batch queue with 120m time limit.\n\n"
            printf "Submit the resulting script with the \`bsub\` command.\n"
            exit 1
            ;;
        -*|--*)
            echo "generate_lassen_scaling_job_script: Unknown option $1"
            exit 1
            ;;
        *)
            NONOPT_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${NONOPT_ARGS[@]}"


NUM_NODES=${NUM_NODES:-"1"}
NUM_PROCS_MAX=$(( 4 * ${NUM_NODES} ))
NUM_PROCS=${NUM_PROCS:-"${NUM_PROCS_MAX}"}
NUM_PROCS_1=${NUM_PROCS_1:-"1"}
OUTPUT_PATH=${OUTPUT_PATH:-"scaling_test_lassen.bsub.sh"}
QUEUE_NAME=${QUEUE_NAME:-"pbatch"}
TIME_LIMIT=${TIME_LIMIT:-"120"}

rm -f ${OUTPUT_PATH}
cat <<EOF > ${OUTPUT_PATH}
#!/bin/bash

#BSUB -nnodes ${NUM_NODES}
#BSUB -G uiuc
#BSUB -W ${TIME_LIMIT}
#BSUB -J scale${NUM_PROCS}
#BSUB -q ${QUEUE_NAME}
#BSUB -o scal${NUM_PROCS}.txt

source ../emirge/config/activate_env.sh
source ../emirge/mirgecom/scripts/mirge-testing-env.sh
source ../scripts/multi_scalability.sh -p ../ -s ${NUM_PROCS_1} -n ${NUM_PROCS}

EOF
