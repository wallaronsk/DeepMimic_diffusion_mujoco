# DiffMimic

This is Kenji's final year project with diffusion models. Project direction is not clear yet.

Code was originally forked from DeepMimic, I have since deleted all unused code.
Main code to pay attention to is in the diffusion folder.

## Quickstart

### Setup

Setup the conda environment with the following command:

```bash
conda env create -f environment.yml

conda activate diffmimic
```

### Running the main motion generation code

All the relevant code is kept within `temporal_unet_diffusion.ipynb`. I have left headers and comments in the notebook to explain what each cell does.

### Loading motion datasets

One key thing to mention about motion datasets, is that the number of frames we load must be a multiple of 8 because of how the U-Net works. So if you have a spinkick motion with 78 frames, we can only load 72 frames. The `motion_dataset` loader automatically chooses the maximum number of frames for you.

### Playing the generated motions

I have included a script to play the generated motions. To use it, run the following command:

```bash
python3 mocap_player.py logs/{exp_name}/sampled_motions/motion1.npy

eg: python3 mocap_player.py logs/walk-motion/sampled_motions/motion1.npy
```

### Sampled with constraints

We can apply conditioning directly in the `models/helpers.py` file under the `apply_conditioning` function. Right now its just hardcoded to change the joint positions to look like the human model is holding a box. But essentially you can change these position tensors to anything you want.

In the joint configs, index 13-15 and 17-19 are shoulders representing an euler tuple, 16 and 20 are elbows representing a scalar rotation value in radians. 

### Evaluating constrained sampling

View original data
```bash
python3 utils/mocap_v2.py
```

View sampled motion
```bash
python3 mocap_player.py logs/test-constrained-sampling-holding-a-box/sampled_motions/motion1.npy
```