#!/bin/bash

echo "Running workflow"

# Building mesh
echo "Making mesh"
cd mesh
./make_mesh.sh
cd ..


# step 1 
echo "Step 1"
cd step1
./run.sh > mirge-0.out
cd ..

# step 2
echo "Step 2"
cd step2
mkdir -p init_data
cp ../step1/restart_data/prediction-000000100-* init_data/.
./run.sh > mirge-0.out
#make_viz
cd ..

# step 3
echo "Step 3"
cd step3
mkdir -p init_data
cp ../step2/restart_data/prediction-000000200-* init_data/.
./run_init.sh > init.out
cp restart_data/step3_init-* init_data/.
./transfer_n0_to_n7.sh > init.out
./run.sh > mirge-0.out
cd ..

