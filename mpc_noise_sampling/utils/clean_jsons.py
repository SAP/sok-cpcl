"""
Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0
"""

import sys
import json

if __name__ == "__main__":
    void_dict = {}
    for file in sys.argv[1:]:
        with open(file,"w") as f:
            json.dump(void_dict,f)