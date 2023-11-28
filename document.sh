#!/usr/bin/env bash

# Creates documentation in HTML and PDF with Sphinx.

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2023

set -uexo pipefail 

script_path=`readlink -f "${BASH_SOURCE[0]}"`
script_dir=`dirname "$script_path"`

mkdir -p "$script_dir"/docs/source/_static
mkdir -p "$script_dir"/docs/source/_templates
cd "$script_dir"/docs
make clean
cd "$script_dir"
sphinx-apidoc -f -o "$script_dir"/docs/source/ "$script_dir" 
cd "$script_dir"/docs
make html

make latexpdf
cd "$script_dir"
deactivate
set +uexo

