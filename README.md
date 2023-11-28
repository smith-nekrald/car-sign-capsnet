# Instructions

Prerequisites:

```
   conda install gh --channel conda-forge
   conda install -c menpo wget
   conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
   conda install -c conda-forge tensorboard
   conda install -c anaconda pillow
   conda install -c conda-forge torch-optimizer
   conda install -c conda-forge kaggle
   conda install -c conda-forge lime
   conda install -c conda-forge shap
   conda install scikit-image
   conda install openpyxl
   pip install -U pyinstaller	
```
   
It can be easier to install them into a separate
conda environment: 

```
    conda create -n torch
    conda activate torch
    <install libraries>
    <run the script>
    conda deactivate 
```
    
Reproduction:

1. Clone repository to the folder `project_root`. 
Rename the project directory such that all python 
files are located in `project_root/capsnet`.
2. Download benchmarks from  
https://drive.google.com/file/d/10Nl3ucj1b4u8H-yvl3JNTIBWLtiVN2Eu/view?usp=sharing
3. Unpack the contents in the folder `project_root`. 
The archived folder `benchmarks` should come right into
the directory `project_root`. I.e. the folder tree should look 
`project_root/benchmarks/{belgium-TSC,china-TSRD,german-GTRSD,rtsd-r1}`.
3. Navigate to `project_root/capsnet`. Call `python main.py` and wait till
execution is over. Results will be stored in the folder 
`project_root/capsnet/traindir`.
4. One can open traindir with TensorBoard by
calling `tensorboard --logdir=./traindir`
from `project_root/capsnet`. Execution summary
is available at `project_root/capsnet/traindir/stats.json`.

Traindir with execution results is available at: 
https://drive.google.com/file/d/1enFqS_TE7eQ5wB3AKLoJ3sTtjVFiuKgP
 
All benchmarks in one place can be found at: 
https://drive.google.com/file/d/10Nl3ucj1b4u8H-yvl3JNTIBWLtiVN2Eu

