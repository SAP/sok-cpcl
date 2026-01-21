#!/bin/bash

#Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#SPDX-License-Identifier: Apache-2.0

program_names=("dgauss_approx skellam laplace_its box_muller")
#program_names=("skellam")
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
    python utils/parser.py -n 3 -p malicious_shamir -i 10 -e "$program" -b $benchmarks_folder -f $benchmarks_file
    python utils/parser.py -n 3 -p shamir -i 10 -e "$program" -b $benchmarks_folder -f $benchmarks_file
    python utils/parser.py -n 2 -p mascot -i 10 -e "$program" -b $benchmarks_folder -f $benchmarks_file
done


for program in "${program_names[@]}"; do
    python utils/stats.py -n 3 -p malicious_shamir -e "$program" -s $stats_file -f $benchmarks_file
    python utils/stats.py -n 3 -p shamir -e "$program" -s $stats_file -f $benchmarks_file
    python utils/stats.py -n 2 -p mascot -e "$program" -s $stats_file -f $benchmarks_file
done