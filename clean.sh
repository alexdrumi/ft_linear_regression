#!/bin/bash


files=("src/thetas.txt" "data/testing_data.csv" "data/training_data.csv")


for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file"
        rm "$file"
    else
        echo "$file does not exist"
    fi
done

echo "Cleanup done."
