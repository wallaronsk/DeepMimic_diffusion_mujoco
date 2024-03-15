import torch

# identity function for conditioning, ie no conditioning
# comment and uncomment the 2 versions to turn conditioning on and off 
def apply_conditioning(x):
    return x


# def apply_conditioning(x):
#     # Make it look like its holding a box
#     elbow_val = 1.57  # for 90degrees
#     shoulder_val = [0.0] * 3
#     shoulder_val = torch.tensor(shoulder_val, dtype=torch.float32)
#     x[:, :, 13:16] = shoulder_val
#     x[:, :, 16] = elbow_val
#     x[:, :, 17:20] = shoulder_val
#     x[:, :, 20] = elbow_val
#     return x
