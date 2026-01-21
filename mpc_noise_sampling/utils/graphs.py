"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import statistics

def plot_data(data, protocol, experiment):
    # iterate over all the stats present in the data
    for stat in data[experiment][protocol]["party_0"]["online"].keys():
        df_online = pd.DataFrame({
            "Party": ["Party 0"] * len(data[experiment][protocol]["party_0"]["online"][stat]) +
                     ["Party 1"] * len(data[experiment][protocol]["party_1"]["online"][stat]),
            "Type": ["Online"] * len(data[experiment][protocol]["party_0"]["online"][stat]) +
                    ["Online"] * len(data[experiment][protocol]["party_1"]["online"][stat]),
            f"{stat.title()}": data[experiment][protocol]["party_0"]["online"][stat] +
                               data[experiment][protocol]["party_1"]["online"][stat]
        })

        df_offline = pd.DataFrame({
            "Party": ["Party 0"] * len(data[experiment][protocol]["party_0"]["offline"][stat]) +
                     ["Party 1"] * len(data[experiment][protocol]["party_1"]["offline"][stat]),
            "Type": ["Offline"] * len(data[experiment][protocol]["party_0"]["offline"][stat]) +
                    ["Offline"] * len(data[experiment][protocol]["party_1"]["offline"][stat]),
            f"{stat.title()}": data[experiment][protocol]["party_0"]["offline"][stat] +
                               data[experiment][protocol]["party_1"]["offline"][stat]
        })

        df = pd.concat([df_online, df_offline])

        # Create the bar plot with confidence intervals
        plt.figure(figsize=(8, 6))
        plt.yscale("log")
        
        #sns.boxplot(data=df, x="Type", y=f"{stat.title()}", hue="Party")#, errorbar="sd")
        ax =sns.barplot(data=df, x="Type", y=f"{stat.title()}", hue="Party", errorbar="sd")
        #ax.bar_label(ax.containers[0], fontsize=10)
        plt.title(f"{stat.title()} for {protocol} Protocol in {experiment} Experiment")
        plt.show()



def main(args):
    with open(args.file_name,"r") as f:
        data = json.load(f)
    plot_data(data, args.protocol, args.experiment)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_name', type=str, required=True, help="json file with the stats")
    parser.add_argument('-p', '--protocol', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-s', '--stats_file_name', type=str, required=True)
    parser.add_argument('-n', '--number_of_parties', type=int, required=True)
    args = parser.parse_args()

    main(args)