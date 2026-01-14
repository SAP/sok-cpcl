import re
import json
import argparse
import os

def parse_log_file(filename):
    data = {}

    with open(filename) as file:
        content = file.read()

        # Extract party and number of parties
        #cmd_line = re.search("Command line: .+ -N (\d+) -p (\d+)", content)
        #data["number of parties:"] = int(cmd_line.group(1))
        #data["party:"] = int(cmd_line.group(2))
        if "terminate called" in content:
            return None
        if "No such file or directory" in content:
            return None
        if "failed" in content:
            return None
        

        regex_patterns = {
            #"online": r"5 threads spent a total of (\d+\.\d+) seconds \((\d+\.\d+) MB, (\d+) rounds\)",
            "online": r"Spent (\d+\.\d+) seconds \((\d+\.\d+) MB, (\d+) rounds\)",
            "offline": r".* (\d*\.\d*) seconds \((\d+\.\d+) MB, (\d+) rounds\)",
            "time:": r"Time = (.+) seconds",
            "cpu time:": r"CPU time = (.+)",# in case of multithreading (overall core time)",
            #"cpu time:": r"CPU time = (.+) (overall core time)",
            "coordination time:": r"Coordination took (.+) seconds",
            "data sent (MB):": r"Data sent = (.+) MB",
            "global data sent (MB):": r"Global data sent = (.+) MB",
            "triples": r"(\d+) * Triples\n",
            "bits": r"(\d+) * Bits\n(.+)",
            "input_tuples": r"(\d+) * Input tuples .+ (.+)"
        }

        for key, pattern in regex_patterns.items():
            match = re.search(pattern, content)
            if match:
                if key in ["online", "offline"]:
                    data[key] = {
                        "time": float(match.group(1)),
                        "bandwidth": float(match.group(2)),
                        "rounds": int(match.group(3))
                    }
                    print(key,data[key])
                else:
                    data[key] = float(match.group(1))

    return data

# python parser.py -n 2 -p mascot -i 10 -e box_muller
# where -n is the number of parties
#       -p is the protocol name
#       -i is the number of iterations 
#       -e is the experiment name
# 
# filenames: benchmarks_p0/mascot_box_muller_p0_i with i from 0 to number of iterations
# open json file "benchmarks.json" which is like the following
# protocols name: {
#   party_0: {
#       "all_info according to function parse log file" : {appends the values from the file benchmarks_p0/mascot_p0_i}
#       }} 
#  and so on for the various parties

#logs = {}
#logs[0] = parse_log_file(filename ="logs/clipping-p0")
#logs[1] = parse_log_file(filename ="logs/clipping-p1")
#
#json_output = json.dumps(logs, indent=4)
#print(json_output)


def main(args):
    with open(args.benchmark_file_name, "r") as f:
        results = json.load(f)
    
    for party in range(args.number_of_parties):
        n_iterations_p = args.number_of_iterations
        iteration = 0
        while iteration < n_iterations_p:
        #for iteration = 0,  iteration in range(n_iterations_p):
            iteration += 1
            print(iteration, n_iterations_p)
            filename = f'{args.base_path}/benchmarks_p{party}/{args.protocol}_{args.experiment}_p{party}_{iteration}'
            print(filename)
            try:
                data = parse_log_file(filename)
            except (FileNotFoundError, FileExistsError):
                continue
            if data is None:
                n_iterations_p +=1
                continue

            results.setdefault(args.experiment, {}).setdefault(args.protocol, {})
            results[args.experiment].setdefault(args.protocol, {}).setdefault(f'party_{party}', {})
            for key, value in data.items():
                if key in ["online", "offline"]:
                    results[args.experiment][args.protocol][f'party_{party}'].setdefault(key, {})
                    for subkey, subvalue in value.items():
                        results[args.experiment][args.protocol][f'party_{party}'][key].setdefault(subkey, []).append(subvalue)
                else:
                    results[args.experiment][args.protocol][f'party_{party}'].setdefault(key, []).append(value)

    # Save to benchmarks.json file
    with open(args.benchmark_file_name, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_of_parties', type=int, required=True)
    parser.add_argument('-p', '--protocol', type=str, required=True)
    parser.add_argument('-i', '--number_of_iterations', type=int, required=True)
    parser.add_argument('-e', '--experiment', type=str, required=True)
    parser.add_argument('-b', '--base_path', type=str, required=True, help="path for the benchmark folder")
    parser.add_argument('-f', '--benchmark_file_name', type=str, required=True)
    args = parser.parse_args()

    main(args)