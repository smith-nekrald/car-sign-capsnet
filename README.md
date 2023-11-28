# Capsule Network for Road Sign Classification

## Description

This project is focused on the problem of traffic sign recognition, where it adapts
and tunes the Capsule Network architecture for this particular classification task.
The resulting architecture is evaluated on different benchmarks and explained with
the LIME approach of Explainable AI. 

## Instructions

Reproduction:

1. Clone the repository. Let us call to the root folder `project_root`, i.e.
all python files are located in `project_root/capsnet`.
2. Download benchmarks from  
https://drive.google.com/file/d/10Nl3ucj1b4u8H-yvl3JNTIBWLtiVN2Eu/view?usp=sharing
3. Unpack the contents in the folder `project_root`. 
The archived folder `benchmarks` should come right into
the directory `project_root`. I.e. the folder tree should look 
`project_root/benchmarks/{belgium-TSC,china-TSRD,german-GTRSD,rtsd-r1}`.
4. Install prerequisites. For Anaconda under Windows, an option is to run the script
`install.bat` inside `project_root/capsnet`.
5. Navigate to `project_root/capsnet`. Call 
`python run_capsnet.py --benchmark all` 
(alternatively, `launch.bat`) and wait until
execution is over. Results will be stored in 
the folder `project_root/capsnet/traindir`.
6. One can open traindir with TensorBoard by
calling `tensorboard --logdir=./traindir`
from `project_root/capsnet`. Execution summary is 
available at `project_root/capsnet/traindir/stats.json`.

Already processed traindir with execution results is available at: 
https://drive.google.com/file/d/1enFqS_TE7eQ5wB3AKLoJ3sTtjVFiuKgP
 
Other functionality:
1. Script `executable.bat` prepares standalone executable.
2. Script `document.bat` prepares Sphinx-processed documentation.
