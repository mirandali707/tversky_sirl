from sklearn.decomposition import PCA
from sirl import load_sirl
import torch

def identity(trajs):
    """
    no encoding, pass thru - return trajs, input_dim
    """
    return trajs, trajs.shape[1]


def ground_truth_feats(trajs, feats):
    """
    2D ground truth feature vector (laptop, upright)
    """
    return feats, feats.shape[1]
    

def pca(trajs, latent_dim):
    """
    pca to latent_dim
    """
    pca = PCA(n_components=latent_dim)
    embeds = pca.fit_transform(trajs) # (N, latent_dim)
    return embeds, latent_dim


def sirl(trajs, latent_dim):
    """
    take frozen SIRL encoder from expt 003
    load from sirl_ckpts (copied relevant ckpts into this dir)
    """
    ckpt_path = f"sirl_ckpts/sirl_dim{latent_dim}_seed0.pth"
    print(f"loading ckpt {ckpt_path}")
    model = load_sirl(ckpt_path)
    model.eval()
    embeds = model(torch.tensor(trajs,dtype=torch.float32)).detach().numpy()
    return embeds, latent_dim