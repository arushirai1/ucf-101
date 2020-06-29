#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=4              # Two cores per task
#SBATCH --ntasks=2                     # But only one task
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --gres=gpu:1                   # And two GPU
#SBATCH --time=12:00:00              # Run for 7 days, at most
#SBATCH --job-name=eval-200k              # Name the job so I can see it in squeue
#SBATCH --output=vrd-%J.out
#SBATCH --mail-type=BEGIN,END,FAIL     # Send me email for various states
#SBATCH --mail-user arai4@hawk.iit.edu        # Use this address

source activate ml

python eval_linear_i3d.py --model ./i3d-deepcluster/checkpoint.pth.tar --conv -1 --lr 0.001 --wd -7 --verbose --exp ./test --workers 16 --batch 12
