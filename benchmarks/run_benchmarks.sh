#!/bin/bash

# Check for at least 2 arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <epochs> 'sample_sizes' [rounds]"
    exit 1
fi



# Assign command line arguments to variables
epochs=$1
sample_sizes=($2) # Sample sizes are passed in quotes, e.g., "20 50 100"

# If rounds is not provided, default to 1
rounds=${3:-1}

# Create the results directory if it does not exist
mkdir -p "results"

# Main loop to run the specified number of rounds
for (( r=1; r<=rounds; r++ )); do
    # Generate a timestamp for the current time
    # For example, YYYYMMDD_HHMMSS format
    stamp=$(date "+%Y%m%d_%H%M%S")
    echo "Round $r of $rounds"
    for samples in "${sample_sizes[@]}"; do
        echo "Running binary_classifier.mojo with ${samples} samples for ${epochs} training epochs."
        mojo_output="results/benchmark_mojo_${stamp}.csv"
        mojo ../binary_classifier.mojo $epochs $samples --silent --csv $mojo_output
    done

    echo " "
    # Loop through the sample sizes and call the Python program with each size
    for samples in "${sample_sizes[@]}"; do
        echo "Running binary_classifier.py with ${samples} samples for ${epochs} training epochs."
        python_output="results/benchmark_py_${stamp}.csv"
        python ../binary_classifier.py --epochs=$epochs --samples=$samples --silent --csv --csv_file_path=$python_output
    done

    echo " "
done

echo "All runs completed."
