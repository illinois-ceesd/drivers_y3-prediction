#!/bin/bash

# . scripts/smoke_test.sh <resource file> <top level directory>
#
# Arg1: Testing resource file
# Arg2: Top level path to driver

# Testing resource file sets stuff like:
# PYOPENCL_CTX
# PYOPENCL_TEST
# MIRGE_HOME (path to mirgecom installation)
# MIRGE_MPI_EXEC (important on Porter and Lassen)
# MIRGE_PARALLEL_SPAWNER (important on Porter and Lassen)
# XDG_CACHE_HOME
NONOPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            TESTING_ENV_RESOURCE="$2"
            shift
            shift
            ;;
        -c|--casename)
            CASENAME_ROOT="$2"
            shift
            shift
            ;;
        -o|--output)
            LOG_PATH="$2"
            shift 
            shift
            ;;
        -p|--path)
            DRIVER_PATH="$2"
            shift
            shift
            ;;
        -*|--*)
            echo "install_mirgecom: Unknown option $1"
            exit 1
            ;;
        *)
            NONOPT_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${NONOPT_ARGS[@]}"

TESTING_ENV_RESOURCE=${TESTING_ENV_RESOURCE:-""}
CASENAME_ROOT=${CASENAME_ROOT:-""}
LOG_PATH=${LOG_PATH:-"log_data"}
DRIVER_PATH=${DRIVER_PATH:-"."}

if [[ ! -z "${CASENAME_ROOT}" ]]; then
    printf "Casename file prefix: ${CASENAME_ROOT}\n"
fi

if [[ ! -z "${TESTING_ENV_RESOURCE}" ]]; then
    printf "Testing environment resource file: ${TESTING_ENV_RESOURCE}\n"
    source ${TESTING_ENV_RESOURCE}
fi

mkdir -p ${LOG_PATH}
cd ${LOG_PATH}
LOG_PATH=$(pwd)
cd -

# Debugging - spew the env to stdout
# printf "MIRGE environment:\n"
# env | grep MIRGE
# env | grep PYOPENCL
# env | grep CACHE
# env | grep CUDA

# Set defaults for these in case they didn't get
# set by the resource file.
# At this level, MPI_EXEC, and PARALLEL_SPAWNER are all that
# are needed to make this script platform-agnostic
MPI_EXEC=${MIRGE_MPI_EXEC:-"mpiexec"}
PARALLEL_SPAWNER=${MIRGE_PARALLEL_SPAWNER:-""}

# path management junk
# Set path to top of driver (assume cwd if unset)
origin=$(pwd)

# Driver scripts need to be run from their home directory to find
# their own resources.
cd ${DRIVER_PATH}
DRIVER_PATH=$(pwd)

printf "Driver directory: ${DRIVER_PATH}\n"

date

# Demonstrate how to run multiple tests
declare -i numfail=0
declare -i numsuccess=0
succeeded_tests=""
failed_tests=""
if [[ ! -z ${CASENAME_ROOT} ]];then
    CASENAME_ROOT="${CASENAME_ROOT}_"
fi

#
# This bit will run each "smoke_test" to generate the timing data
# which will be stuffed into LOG_PATH. Any number of tests should
# be allowed.
# - The casename thing is a "nice to have" as it allows some control
#   to caller for what the sqlite files are named.
#
printf "Running parallel timing tests...\n"
# test_directories="scalability_test"
test_path="scalability_test"
test_name="prediction-scalability"

printf "* Running ${test_name} test in ${test_path}.\n"
running_casename_base="${CASENAME_ROOT}${test_name}"

cd ${DRIVER_PATH}/${test_path}

for nrank in 1 2 4; do

    if [[ "${nrank}" == "1" ]]; then
        msize="48"
        nelem="24036"
    elif [[ "${nrank}" == "2" ]]; then
        msize="30.5"
        nelem="47908"
    elif [[ "${nrank}" == "4" ]]; then
        msize="21.5"
        nelem="96425"
    elif [[ "${nrank}" == "8" ]]; then
        msize="16.05"
        nelem="192658"
    elif [[ "${nrank}" == "16" ]]; then
        msize="12.3"
        nelem="383014"
    fi

    casename="${running_casename_base}_np${nrank}"
    printf "** Running ${casename} on ${nrank} ranks.\n"

    cd data
    rm actii.msh
    ./mkmsh --size=${msize} --nelem=${nelem} --link
    cd ../

    set -x
    $MPI_EXEC -n ${nrank} $PARALLEL_SPAWNER python -u -m mpi4py driver.py -c ${casename} -g ${LOG_PATH} -i run_params.yaml --log --lazy
    return_code=$?
    set +x

    mv viz_data viz_data_${nrank}
    mv restart_data restart_data_${nrank}


    if [[ $return_code -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "** ${test_path}/${casename} $succeeded."
        succeeded_tests="$succeeded_tests ${test_path}/${casename}"
    else
        ((numfail=numfail+1))
        echo "**  ${test_path}/${casename} failed."
        failed_tests="$failed_tests ${test_path}/${casename}"
    fi

done

date
printf "Scaling/timing tests done.\n"
printf "Passing tests: ${succeeded_tests}\n"
printf "Failing tests: ${failed_tests}\n"

cd ${origin}
return $numfail
