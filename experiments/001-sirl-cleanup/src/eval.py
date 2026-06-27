import numpy as np
import torch
from sklearn.linear_model import LinearRegression


def eval_model(config, data, model):
    """
    eval
    """
    print("eval")
    eval_params = config["eval"]
    if eval_params["method"] == "fpe":
        fpe = eval_fpe(config, data, model)
        print(f"fpe: {fpe}")
        return {"fpe": fpe}
    return {}


def eval_fpe(config, data, model):
    train_trajs = data["train_trajs"]
    test_trajs = data["test_trajs"]
    train_features = data["train_features"]
    test_features = data["test_features"]

    # get embeddings (run train, test data through model)
    if config["model"]["name"] == "pca":
        # sklearn pca
        Z_train = model.transform(train_trajs.reshape(len(train_trajs), -1))
        Z_test  = model.transform(test_trajs.reshape(len(test_trajs), -1))
    else:
        # pytorch (e.g. SIRL)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        with torch.no_grad():
            Z_train = model(torch.as_tensor(train_trajs, dtype=torch.float32, device=device)).cpu().numpy()
            Z_test  = model(torch.as_tensor(test_trajs,  dtype=torch.float32, device=device)).cpu().numpy()
    # fit fpe probe on train embeds -> ground truth features
    reg = LinearRegression().fit(Z_train, train_features)
    pred = reg.predict(Z_test) # predict test labels
    fpe = np.mean(np.sum((pred - test_features) ** 2, axis=1)) # report mse
    return fpe