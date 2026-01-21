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



# build_mp_spdz_macos.sh
# Convenience script to prepare macOS dependencies and build MP-SPDZ in this repo.

echo "Starting MP-SPDZ macOS build helper"

# Check if Homebrew is installed
if ! command -v brew >/dev/null 2>&1; then
  echo "Error: Homebrew not found."
  echo "Please install Homebrew first: https://brew.sh/"
  exit 1
fi

echo "✓ Homebrew is installed"

# Check if git is available
if ! command -v git >/dev/null 2>&1; then
  echo "Error: git not found. Installing via Homebrew..."
  brew install git
fi

echo "✓ git is available"

echo "Installing required Homebrew packages (may prompt for password)..."
brew update
brew install gmp openssl@1.1 libsodium cmake automake autoconf libtool pkg-config || true

OPENSSL_PREFIX=$(brew --prefix openssl@1.1 || brew --prefix openssl)
GMP_PREFIX=$(brew --prefix gmp)
SODIUM_PREFIX=$(brew --prefix libsodium)

export LDFLAGS="-L${OPENSSL_PREFIX}/lib -L${GMP_PREFIX}/lib -L${SODIUM_PREFIX}/lib"
export CPPFLAGS="-I${OPENSSL_PREFIX}/include -I${GMP_PREFIX}/include -I${SODIUM_PREFIX}/include"
export PKG_CONFIG_PATH="${OPENSSL_PREFIX}/lib/pkgconfig:${GMP_PREFIX}/lib/pkgconfig:${SODIUM_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo "Environment set:" 
echo " LDFLAGS=${LDFLAGS}"
echo " CPPFLAGS=${CPPFLAGS}"

# Check if MP-SPDZ directory exists
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

