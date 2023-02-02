#!/bin/bash
#SBATCH -A als_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-240:10%4
#SBATCH --output=%x-%a.%j.out

echo "Starting at: $(date)"
module load python
conda activate batteryNET
echo "In conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi
export SLURM_CPU_BIND="cores"
STACK=IM_298-1_$(printf "%03d" $SLURM_ARRAY_TASK_ID)
python predict_2d.py setup_files/xsection-unet2d.json --predict_stack $STACK
echo "Done at : $(date)"
