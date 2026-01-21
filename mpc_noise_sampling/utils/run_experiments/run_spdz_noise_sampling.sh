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




program_names=("laplace_its" "box_muller" "skellam" "dgauss_approx")
protocol_names_2p_f=("mascot")
protocol_names_3p_f=("shamir malicious-shamir-party.x")

#N_value=0
p_value=0
iterations=0
file_path=""

while getopts N:p:i:f: flag
do
    case "${flag}" in
        #N) N_value=${OPTARG};;
        p) p_value=${OPTARG};;
        i) iterations=${OPTARG};;
        f) file_path=${OPTARG};;

    esac
done

# Semi-honest honest majority
for program in "${program_names[@]}"; do
    i=1
    echo "Shamir ${program}"
    while [ $i -le $iterations ]; do
        echo "${i}"
        ./shamir-party.x -N 3 -ip "Player-Data/ip-file.txt" -v -p $p_value "$program" &> "${file_path}/shamir_${program}_p${p_value}_${i}"
        i=$(($i+1))
    done
done

# Malicious honest majority
for program in "${program_names[@]}"; do
    i=1
    echo "Malicious-Shamir ${program}"
    while [ $i -le $iterations ]; do
        echo "${i}"
        ./malicious-shamir-party.x -N 3 -ip "Player-Data/ip-file.txt" -v -p $p_value "$program" &> "${file_path}/malicious_shamir_${program}_p${p_value}_${i}"
        i=$(($i+1))
    done
done

# Malicious dishonest majority
for program in "${program_names[@]}"; do
    i=1
    echo "Mascot ${program}"
    while [ $i -le $iterations ]; do
        echo "${i}"
        ./mascot-party.x -N 2 -ip "Player-Data/ip-file.txt" -v -p $p_value "$program" &> "${file_path}/mascot_${program}_p${p_value}_${i}"
        i=$(($i+1))
    done
done

