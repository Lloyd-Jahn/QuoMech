#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC \
    -Wall -Wextra \
    "$DIR/ode_engine.cpp" -o "$DIR/libquomech_ode.so"
echo "Built $DIR/libquomech_ode.so"
