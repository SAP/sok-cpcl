#!/bin/bash

program_names=("cos" "sin" "exp" "inv_sqrt" "inv" "log" "sqrt")
#N_value=0
p_value=0
iterations=0

while getopts s:f:b:i: flag
do
    case "${flag}" in
        s) stats_file=${OPTARG};;
        f) benchmarks_file=${OPTARG};;
        b) benchmarks_folder=${OPTARG};;
        i) iterations=${OPTARG};;
    esac
done

echo $benchmarks_folder

for program in "${program_names[@]}"; do
    python utils/parser.py -n 2 -p shamir -i 10 -e "$program" -b $benchmarks_folder -f $benchmarks_file
done


for program in "${program_names[@]}"; do
    python utils/stats.py -n 2 -p shamir -e "$program" -s $stats_file -f $benchmarks_file
done