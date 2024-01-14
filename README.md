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

Now you are able to run the notebook, the only one to pay attention to is `temporal_unet_diffusion.ipynb`. I have left headers and comments in the notebook to explain what each cell does.

### Playing the generated motions

I have included a script to play the generated motions. To use it, run the following command:

```bash
python3 mocap_player.py logs/{exp_name}/sampled_motions/motion1.npy

eg: python3 mocap_player.py logs/walk-motion/sampled_motions/motion1.npy
```

