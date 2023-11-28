@echo off

REM Launches experiment on all benchmarks sequentially.

REM Author: Aliaksandr Nekrashevich
REM Email: aliaksandr.nekrashevich@queensu.ca
REM (c) Smith School of Business, 2023

call conda activate torch
call python run_capsnet.py --benchmark all
call conda deactivate

