from sirl import load_sirl
import torch

SIRL_CKPT_DIR = "../001-sirl-cleanup/results/sirl_gridrobot"

def get_sirl(seed, latent_dim):
    ckpt_path = f"{SIRL_CKPT_DIR}/sirl_dim{latent_dim}_seed{seed}.pth"
    sirl = load_sirl(ckpt_path)
    sirl.requires_grad_(False) # no backprop pls
    return sirl
