import sys
import json

if __name__ == "__main__":
    void_dict = {}
    for file in sys.argv[1:]:
        with open(file,"w") as f:
            json.dump(void_dict,f)