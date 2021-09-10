#!/bin/sh
### Set the job name (for your reference)
#PBS -N graphwavenet_test
### Set the project name, your department code by default
#PBS -P scai
### Request email when job begins and ends, don't change anything on the below line 
# -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=1:ngpus=1:mem=24G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=02:00:00
#PBS -l software=PYTHON
#PBS -q low

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

module () {
        eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
}

module load compiler/python/3.6.0/ucs4/gnu/447
module load pythonpackages/3.6.0/ucs4/gnu/447/pip/9.0.1/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/setuptools/34.3.2/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/wheel/0.30.0a0/gnu
module load pythonpackages/3.6.0/ucs4/gnu/447/numpy/1.12.0/intel
module load pythonpackages/3.6.0/pandas/0.23.4/gnu
module load compiler/cuda/9.2/compilervars
module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu
module load apps/pythonpackages/3.6.0/torchvision/0.2.1/gpu
module load compiler/gcc/9.1.0
module load apps/anaconda/3
module load lib/openblas/0.2.20/gnu
module load lib/hdf5/1.8.20/gnu

python3 two_b.py --batch-size 1e6 --epochs 83000 --eps 1e-3
