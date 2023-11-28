@echo off
REM Creates documentation in HTML and PDF with Sphinx.

REM Author: Aliaksandr Nekrashevich
REM Email: aliaksandr.nekrashevich@queensu.ca
REM (c) Smith School of Business, 2023


cd docs
call make.bat clean
cd ..
call sphinx-apidoc -f -o "docs/source/" "./" 
cd docs
call make html
call make latexpdf
cd ..



