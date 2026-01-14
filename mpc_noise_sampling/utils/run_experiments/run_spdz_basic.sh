#!/bin/bash


program_names=("cos" "sin" "exp" "inv_sqrt" "inv" "log" "sqrt")
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

for program in "${program_names[@]}"; do
    i=1
    echo "Mascot ${program}"
    while [ $i -le $iterations ]; do
        echo "${i}"
        i=$(($i+1))
        ./mascot-party.x -N 2 -ip "Player-Data/ip-file.txt" -v -p $p_value "$program" &> "${file_path}/shamir_${program}_p${p_value}_${i}"
    done
done
