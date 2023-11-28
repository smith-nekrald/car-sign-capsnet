@echo off

REM Creates documentation in HTML and PDF with Sphinx.

REM Author: Aliaksandr Nekrashevich
REM Email: aliaksandr.nekrashevich@queensu.ca
REM (c) Smith School of Business, 2023


cd docs
make.bat clean
cd ..
sphinx-apidoc -f -o "docs/source/" "./" 
cd docs
make html

make latexpdf
cd ..

set +uexo

