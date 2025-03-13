#!/bin/bash

# Check if a file path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dmrg_file_path> [custom_reader_module]"
    echo "Example: $0 test/1.C22H12_triangulene_s0/0512_m"
    echo "Example with custom reader: $0 test/1.C22H12_triangulene_s0/0512_m my_reader_module"
    exit 1
fi

DMRG_FILE=$1
CUSTOM_READER=""

# Check if a custom reader module is provided
if [ $# -eq 2 ]; then
    CUSTOM_READER=$2
fi

# Copy the DMRG file to a temporary location to avoid file access issues
TEMP_FILE="./temp_dmrg_file"
cp "$DMRG_FILE" "$TEMP_FILE"

# Build the Docker image if it doesn't exist
docker-compose build

# Run the prediction
if [ -z "$CUSTOM_READER" ]; then
    docker-compose run qc_dmrg_pred "$TEMP_FILE"
else
    docker-compose run qc_dmrg_pred "$TEMP_FILE" "$CUSTOM_READER"
fi

# Clean up the temporary file
rm -f "$TEMP_FILE" 