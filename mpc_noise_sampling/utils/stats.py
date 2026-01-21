"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import json
import argparse
import statistics
import math
import scipy.stats as st
import numpy as np

z = 1.96 # for c_i at 95%
def compute_stats(data, protocol, experiment, stats, n_parties):
    stats.setdefault(experiment, {}).setdefault(protocol, {})

    for party_i in range(n_parties):
        stats[experiment].setdefault(protocol, {}).setdefault(f'party_{party_i}', {})

        for key, value in data[experiment][protocol][f'party_{party_i}'].items():
            if key in ["online", "offline"]:
                print(key)
                stats[experiment][protocol][f'party_{party_i}'].setdefault(key, {})
                for subkey, _ in value.items():
                    mean = np.mean(data[experiment][protocol][f'party_{party_i}'][key][subkey])
                    n = len(data[experiment][protocol][f'party_{party_i}'][key][subkey])
                    std_dev = np.std(data[experiment][protocol][f'party_{party_i}'][key][subkey])
                    ci = st.t.interval(confidence=0.95, df=n-1, loc=mean, scale=st.sem(data[experiment][protocol][f'party_{party_i}'][key][subkey]))

                    stats[experiment][protocol][f'party_{party_i}'][key][subkey] = {
                        "mean": mean,
                        "std_dev": std_dev,
                        "ci (95 %)" :  mean - ci[0]
                    }
            else:
                n = len(data[experiment][protocol][f'party_{party_i}'][key])
                mean = np.mean(data[experiment][protocol][f'party_{party_i}'][key])
                std_dev = np.std(data[experiment][protocol][f'party_{party_i}'][key])
                ci = st.t.interval(confidence=0.95, df=n-1, loc=mean, scale=st.sem(data[experiment][protocol][f'party_{party_i}'][key]))
                stats[experiment][protocol][f'party_{party_i}'][key] = {
                    "mean": mean,
                    "std_dev": std_dev,
                    "ci (95 %)" :  mean - ci[0]
                }
    return stats
    

def main(args):
    with open(args.benchmarks_file_name,"r") as f:
        data = json.load(f)
    with open(args.stats_file_name, "r") as s:
        stats = json.load(s)
        compute_stats(data, args.protocol, args.experiment, stats, args.number_of_parties)
    with open(args.stats_file_name, "w") as s:
        json.dump(stats,s, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--benchmarks_file_name', type=str, required=True, help="json file with the benchmarks")
    parser.add_argument('-p', '--protocol', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-s', '--stats_file_name', type=str, required=True)
    parser.add_argument('-n', '--number_of_parties', type=int, required=True)
    args = parser.parse_args()

    main(args)