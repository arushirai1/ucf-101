# ucf-101

## How to Train DeepCluster with i3d as base on UCF-101 data

1. Change paths to dataset in build_paths function in Utils.py
2. (Not required) Edit sampling method in create_frames function in Utils.py
3. If you are not running on at least 2 gpus, then edit video-i3d-main.py and remove .module everywhere

### Run
This slurm script has an example of arguments: w16.slurm
```
python video-i3d-main.py --resume [path to model you are resuming from] --exp [path to where you want to store model] --k [number of clusters] --workers [number of workers] --verbose
```

## Evaluate on Downstream task
Check out i3d_eval.slurm for example arguments to run eval_linear_i3d.py

** If you are running on COIN check out https://github.com/arushirai1/COIN_HowTo and just edit the --model in the eval_COIN_120.slurm script and edit paths in that Utils.py folder
