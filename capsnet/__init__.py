""" This module implements Car Sign recognition with Capsule Network. """

# Author: Aliaksandr Nekrashevich
# Email: aliaksandr.nekrashevich@queensu.ca
# (c) Smith School of Business, 2023


import sys
import pathlib

_file_path: pathlib.Path = pathlib.Path(__file__)
sys.path.append(str(_file_path.absolute().parent))
sys.path.append(str(_file_path.absolute().parent.parent))
