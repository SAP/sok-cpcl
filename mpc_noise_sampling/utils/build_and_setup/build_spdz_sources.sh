#!/bin/bash


cp src/spdz/* ~/mpc/mp-spdz/Programs/Source/
cp data/Player-Data/* ~/mpc/mp-spdz/Player-Data/

program_names=("laplace_its" "box_muller" "skellam" "d_gauss" "cos" "clipping" "sin" "exp" "inv_sqrt" "inv" "log" "sqrt" "dgauss_approx")
protocol_names_2p_f=("mascot")
protocol_names_3p_f=("shamir malicious-shamir-party.x")

make -j8 mascot-party.x
make -j8 shamir-party.x
make -j8 malicious-shamir-party.x

for program in "${program_names[@]}"; do
    ./compile.py -F 64 "$program"
done