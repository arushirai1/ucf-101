#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=4              # Two cores per task
#SBATCH --ntasks=2                     # But only one task
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --gres=gpu:2                   # And two GPU
#SBATCH --time=16:00:00              # Run for 7 days, at most
#SBATCH --job-name=deepcluster-i3d              # Name the job so I can see it in squeue
#SBATCH --output=vrd-%J.out
#SBATCH --mail-type=BEGIN,END,FAIL     # Send me email for various states
#SBATCH --mail-user arai4@hawk.iit.edu        # Use this address

module list
conda env list
source activate ml

python video-i3d-main.py --resume i3d-deepcluster/checkpoint.pth.tar --exp ./i3d-deepcluster --k 200 --workers 16 --verbose
