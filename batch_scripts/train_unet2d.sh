#!/bin/bash
#SBATCH -A als_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --output=%x.%j.out

echo "Starting at: $(date)"
module load python
# module load pytorch/1.10.0
conda activate batteryNET
echo "In conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi
export SLURM_CPU_BIND="cores"
python train.py 'setup_files/xsection-unet2d.json' --num_workers 32 --epochs 100
STACK=IM_298-1_240
python predict_2d.py 'setup_files/xsection-unet2d.json' --predict_stack $STACK
echo "Done at : $(date)"
