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

printf "Testing resource file: ${1}\n"
if [[ -z "${1}" ]]; then
    source ${1}
fi

# Debugging - spew the env to stdout
# printf "MIRGE environment:\n"
# env | grep MIRGE
# env | grep PYOPENCL
# env | grep CACHE
# env | grep CUDA

# Set defaults for these in case they didn't get
# set by the resource file.
MPI_EXEC=${MIRGE_MPI_EXEC:-"mpiexec"}
PARALLEL_SPAWNER=${MIRGE_PARALLEL_SPAWNER:-""}

# path management junk
# Set path to top of driver (assume cwd if unset)
TOP_PATH=${2:-"."}
origin=$(pwd)
cd ${TOP_PATH}
TOP_PATH=$(pwd)
printf "Driver directory: ${TOP_PATH}\n"

python -m pip install -e .

date

# Demonstrate how to run multiple tests
declare -i numfail=0
declare -i numsuccess=0
succeeded_tests=""
failed_tests=""

printf "Running serial tests...\n"
# serial_test_names="smoke_test smoke_test_3d smoke_test_ks"
serial_test_names="smoke_test smoke_test_ks"
for test_name in $serial_test_names
do
    test_path=${test_name}
    printf "* Running test ${test_name} in ${test_path}\n"
    cd ${TOP_PATH}/${test_path}

    # Create 3d mesh if not already there
    if [[ "${test_name}" == *"_3d"* ]]; then
        cd data
        ./mkmsh --size=48 --nelem=24110 --link  # will not overwrite if exists
        cd ../
    fi

    $MPI_EXEC -n 1 $PARALLEL_SPAWNER python -u -m mpi4py driver.py -i run_params.yaml --log --lazy
    return_code=$?
    cd -

    if [[ $return_code -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "** ${test_name} succeeded."
        succeeded_tests="$succeeded_tests ${test_name}"
    else
        ((numfail=numfail+1))
        echo "** Example $example failed."
        failed_tests="$failed_tests ${test_name}"
    fi

done

date
printf "Serial tests done.\n"
printf "Running parallel tests.\n"

parallel_test_names="smoke_test_ks"
for test_name in $parallel_test_names
do
    test_path=${test_name}
    test_name="parallel_${test_name}"
    printf "* Running test ${test_name} in ${test_name}."
    cd ${TOP_PATH}/${test_path}
    
    # Create 3d mesh unless already there
<<<<<<< HEAD
    cd data
    ./mkmsh --size=30.5 --nelem=47908 --link  # will not overwrite if it exists
    cd ../
=======
    if [ "${test_name}" == "smoke_test_ks_3d" ]; then
        cd data 
        rm actii.msh
        if [[ -f "actii_47908.msh" ]]; then
            ln -s actii_47908.msh actii.msh
        else
            ./mkmsh --size=30.5 --link  # will not overwrite if it exists
        fi
        cd ../
    fi
>>>>>>> main

    # Run the case
    $MPI_EXEC -n 2 $PARALLEL_SPAWNER python -u -m mpi4py driver.py -i run_params.yaml --log --lazy
    return_code=$?
    cd -

    if [[ $return_code -eq 0 ]]
    then
        ((numsuccess=numsuccess+1))
        echo "** Test ${test_name} succeeded."
        succeeded_tests="$succeeded_tests ${test_name}"
    else
        ((numfail=numfail+1))
        echo "** Test ${test_name} failed."
        failed_tests="$failed_tests ${test_name}"
    fi
done

printf "Passing tests: ${succeeded_tests}\n"
printf "Failing tests: ${failed_tests}\n"

cd ${origin}
return $numfail
