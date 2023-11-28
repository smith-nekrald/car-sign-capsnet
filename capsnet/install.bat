@echo off

REM Creates environment and installs relevant libraries.

REM Author: Aliaksandr Nekrashevich
REM Email: aliaksandr.nekrashevich@queensu.ca
REM (c) Smith School of Business, 2023

call conda create -n torch
call conda activate torch

call conda install gh --channel conda-forge
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
call conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
call conda install -c conda-forge tensorboard
call conda install -c anaconda pillow
call conda install -c conda-forge kaggle
call conda install -c conda-forge lime
call conda install scikit-image
call conda install openpyxl
call pip install -U pyinstaller
call conda install -c anaconda sphinx
call conda install -c conda-forge perl

call conda deactivate
