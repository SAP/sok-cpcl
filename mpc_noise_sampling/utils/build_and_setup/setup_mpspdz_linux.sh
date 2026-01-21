#!/usr/bin/env bash

#Copyright (c) 2026 SAP SE or an SAP affiliate company and sok-cpcl contributors
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# build_mp_spdz_linux.sh
# Convenience script to prepare Linux (Debian/Ubuntu) dependencies and build MP-SPDZ in this repo.

echo "Starting MP-SPDZ Linux build helper"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found. This script is designed for Debian/Ubuntu-based distributions."
  echo "For other distributions, install: build-essential, g++, git, python3, python3-dev, libgmp-dev, libssl-dev, libsodium-dev, cmake, automake, autoconf, libtool, pkg-config"
  exit 1
fi

echo "Updating package manager and installing required dependencies (may prompt for password)..."
sudo apt-get update
sudo apt-get install -y build-essential g++ git python3 python3-dev python3-pip libgmp-dev libssl-dev libsodium-dev cmake automake autoconf libtool pkg-config

echo "Checking for optional boost development package..."
sudo apt-get install -y libboost-all-dev || echo "Note: libboost-all-dev is optional and may not be available in all distributions."

if [ ! -d "MP-SPDZ" ]; then
  echo "MP-SPDZ directory not found in current folder. Cloning..."
  git clone https://github.com/data61/MP-SPDZ.git
  cd MP-SPDZ
  Scripts/tldr.sh
  Scripts/setup-ssl.sh 16
  echo "✓ MP-SPDZ cloned successfully"
else
  echo "✓ MP-SPDZ directory found"
fi