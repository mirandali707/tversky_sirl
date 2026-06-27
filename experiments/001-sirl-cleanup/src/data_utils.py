import numpy as np

def load_data(config):
    """
    loads simulated data from .npz
    data object has "anchors", "positives", "negatives" keys
    each is a np array with shape (n_triplets, 19) since 19 is gridrobot trajectory dim (9 xy coords + one joint angle)
    """
    data_params = config["data"]
    print(f"loading data: {data_params["dataset_name"]}")
    # TODO
    # np.load(data_params.path)
