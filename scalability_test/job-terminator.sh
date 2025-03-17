#!/bin/bash
#
# job-terminator.sh
#
# Usage:
#   ./job-terminator <jobid>
#
# Given a Flux job ID, this script:
# 1. Extracts the node list from `flux jobs`.
# 2. Displays the nodes and asks for confirmation.
# 3. SSHs into each node and force-kills (-9) all processes owned by the user.

if [ -z "$1" ]; then
    echo "Usage: $0 <jobid>"
    exit 1
fi

jobid="$1"

# Get the line matching the job ID
job_line=$(flux jobs | grep "$jobid ")

if [ -z "$job_line" ]; then
    echo "Error: Job ID $jobid not found."
    exit 1
fi

# Extract the nodelist (last field)
nodelist=$(echo "$job_line" | awk '{print $NF}')

# Extract the common prefix and the numeric ranges
prefix=$(echo "$nodelist" | sed -E 's/^([^[]+)\[.*$/\1/')
ranges=$(echo "$nodelist" | sed -E 's/^[^[]+\[(.*)\]$/\1/')

# Expand the ranges into individual node names
IFS=',' read -ra parts <<< "$ranges"
nodes=()

for part in "${parts[@]}"; do
    if [[ "$part" == *"-"* ]]; then
        start=$(echo "$part" | cut -d'-' -f1)
        end=$(echo "$part" | cut -d'-' -f2)
        for (( i=start; i<=end; i++ )); do
            nodes+=( "${prefix}${i}" )
        done
    else
        nodes+=( "${prefix}${part}" )
    fi
done

# Display nodes and ask for confirmation
echo "The following nodes are running job $jobid:"
printf "  %s\n" "${nodes[@]}"
echo
read -p "Are you sure you want to dispatchinate the job on these nodes? [Y/n]: " confirm

if [[ "$confirm" =~ ^[Nn]$ ]]; then
    echo "Aborting."
    exit 0
fi

# Loop through each node and SSH to kill processes
for node in "${nodes[@]}"; do
    echo "Dispatchinating node: $node..."
    ssh "$node" "pkill -9 -u \$USER" && echo "Successfully dispatchinated $node" || echo "Failed to dispatchinate $node"
done

echo "Dispatchination complete."
