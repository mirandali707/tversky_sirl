import itertools
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LinearRegression
from tversky_utils import retrieve_semantic_expression
from sklearn.manifold import TSNE
import plotly.express as px
# from encoders import *

# raw feature columns in data["features"], in order (matches make_tversky_report.py)
FEATURE_NAMES = ["laptop (computer_dist)", "upright (joint_up)"]


def eval_model(config, data, model):
    """
    eval
    """
    print("eval")
    eval_params = config["eval"]
    if isinstance(eval_params, dict):  # single method given as a bare dict
        eval_params = [eval_params]
    all_eval = {}
    for entry in eval_params:
        method = entry["method"]
        if method == "fpe":
            fpe = eval_fpe(config, data, model)
            print(f"fpe: {fpe}")
            all_eval = all_eval | {"fpe": fpe}
        if method == "tpa":
            tpa = eval_tpa(config, data, model)
            print(f"tpa: {tpa}")
            tpa_dict = {
                "tpa": tpa,
                "tpa_mean": np.mean(tpa),
                "tpa_std": np.std(tpa)
            }
            all_eval = all_eval | tpa_dict
        if method == "query":
            query = eval_queries(config, data, model, entry)   # {bank: {feature: ...}}
            print("QUERY RESULTS")
            for bank_name, bank_query in query.items():        # "proj" and/or "sim"
                for f_name in FEATURE_NAMES:
                    f_query = bank_query[f_name]
                    print(f"[{bank_name}] {f_name}: n_pairs {f_query['n_pairs']}, "
                          f"n_ttest_run {f_query['n_ttest_run']}, "
                          f"n_significant {f_query['n_significant']}")
                all_eval[f"query_{bank_name}"] = bank_query
            all_eval = all_eval | {"query": query}
        if method == "tsne":
            results_dir = tversky_sim_tsne(config, data, model)
            print(f"figures saved into {results_dir}")
    return all_eval


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


def eval_tpa(config, data, model):
    pref_pairs  = data["pref_pairs"]   # (n_pref, 2) - indices into trajs
    trajs       = data["trajs"]        # (n_traj, ...) - trajectory pool the pairs index into
    # one preference label set per theta: pref_labels_0, pref_labels_1, ...
    label_keys  = sorted(k for k in data.keys() if k.startswith("pref_labels_"))

    # get embeddings (run all trajs through model) -- same branch as eval_fpe
    if config["model"]["name"] == "pca":
        Z = model.transform(trajs.reshape(len(trajs), -1))
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()
        with torch.no_grad():
            Z = model(torch.as_tensor(trajs, dtype=torch.float32, device=device)).cpu().numpy()

    # train/test split over PREFERENCE PAIRS (the split that must be clean for tpa).
    # pref_pairs are shared across thetas, so the split is the same for every set.
    rng = np.random.default_rng(config.get("seed", 0))
    n = len(pref_pairs)
    perm = rng.permutation(n)
    n_test = int(round(config.get("tpa_test_frac", 0.2) * n))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    train_pairs, test_pairs = pref_pairs[train_idx], pref_pairs[test_idx]

    # torch tensors for the reward head (embeddings are frozen, like FPE's probe)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Z = torch.as_tensor(Z, dtype=torch.float32, device=device)
    train_pairs = torch.as_tensor(train_pairs, dtype=torch.long, device=device)
    test_pairs  = torch.as_tensor(test_pairs,  dtype=torch.long, device=device)

    tpas = []
    for key in label_keys:
        pref_labels  = data[key]
        train_labels = torch.as_tensor(pref_labels[train_idx], dtype=torch.float32, device=device)
        test_labels  = torch.as_tensor(pref_labels[test_idx],  dtype=torch.long,    device=device)

        # reward head R_theta on top of frozen embeddings
        hidden, n_layers = config.get("tpa_hidden", 128), config.get("tpa_layers", 2)
        layers, in_dim = [], Z.shape[1]
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]; in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        reward = nn.Sequential(*layers).to(device)
        opt = torch.optim.Adam(reward.parameters(), lr=config.get("tpa_lr", 1e-3),
                               weight_decay=config.get("tpa_l2", 0.0))
        bce = nn.BCEWithLogitsLoss()

        # Bradley-Terry: P(A>B) = sigmoid(R(A) - R(B))  (Eq. 3); BCE on logits = Eq. (4)
        # label 0 => A preferred => target prob(A>B) = 1; label 1 => target = 0  => y = 1 - label
        A_tr, B_tr = train_pairs[:, 0], train_pairs[:, 1]
        y_tr = 1.0 - train_labels
        batch_size = config.get("tpa_batch_size", 64)
        reward.train()
        for _ in range(config.get("tpa_epochs", 500)):
            bperm = torch.randperm(len(train_pairs), device=device)
            for s in range(0, len(bperm), batch_size):
                bi = bperm[s:s + batch_size]
                logits = reward(Z[A_tr[bi]]).squeeze(-1) - reward(Z[B_tr[bi]]).squeeze(-1)
                loss = bce(logits, y_tr[bi])
                opt.zero_grad(); loss.backward(); opt.step()

        # TPA = preference accuracy on held-out pairs
        reward.eval()
        with torch.no_grad():
            rA = reward(Z[test_pairs[:, 0]]).squeeze(-1)
            rB = reward(Z[test_pairs[:, 1]]).squeeze(-1)
            pred = (rA <= rB).long()                 # rA>rB -> A preferred -> label 0
            tpas.append((pred == test_labels).float().mean().item())

    return tpas


def eval_queries(config, data, model, eval_params=None, train_config=None, save_pairs=False):
    """
    * all possible pairs for min / max joint angle, min / max laptop, and see how often non-empty queries with significant t-test of means happens
        * get {set of trajs with max joint angle feature value} {set of trajs with min joint angle feature value}
        * make all pairs dataset
        * for each pair, make 2x queries from trajectories (min - max, max - min) keep track of:
                * num non-empty queries (0, 1, or 2)
                * if 2: compare two samples via t-test of means
                        * report one-sided t-test of means (max - min should be > min - max) stats
                        * flags or value for significant at 0.05, 0.01, .001

    Runs the eval on every Tversky feature bank the model exposes:
        tversky_sirl   -> ["sim"]
        tversky_sirl_2 -> ["sim", "proj"]
        tversky_proj   -> ["proj"]
    Output is keyed by bank label, then feature name: results[bank][feature].

    save_pairs: when False (default), per-pair records are not built or returned.
    """
    all_trajs = data["trajs"]      # (N, 19)
    all_feats = data["features"]   # (N, 2)

    # knobs — read from this method's eval_params entry (e.g. {method: query, max_pairs: 1000})
    q_cfg            = eval_params or {}
    top_feature_count = q_cfg.get("top_feature_count", 4)
    top_result_count = q_cfg.get("top_result_count", 5)
    alphas           = q_cfg.get("alphas", [0.05, 0.01, 0.001])
    max_pairs        = q_cfg.get("max_pairs", None)   # if set, sample this many (max, min) pairs per feature
    rng              = np.random.default_rng(config.get("seed", 0))

    # TODO pass trajs through trained sirl
    # # transform all trajs according to encoder
    # if config["model"]["encoder"] == "pca":
    #     # use PCA embeds as input
    #     latent_dim = config["model"]["latent_dim"]
    #     all_trajs, input_dim = pca(all_trajs, latent_dim)
    # if config["model"]["encoder"] == "sirl":
    #     # use SIRL embeds as input
    #     latent_dim = config["model"]["latent_dim"]
    #     all_trajs, input_dim = sirl(all_trajs, latent_dim)
    # if config["model"]["encoder"] == "feats":
    #     # use ground truth features as input
    #     all_trajs = all_feats
    trajs_t = torch.as_tensor(all_trajs, dtype=torch.float32)

    # --- which Tversky feature banks does this model have? ---
    train_config = train_config or config          # falls back to `config` if it holds ["model"]
    model_name = train_config["model"]["name"]
    # TODO change for just tverskysim
    banks_present = {
        "literally_just_tversky_sim":   ["sim_only"],
        "tversky_sirl":   ["sim"],
        "tversky_sirl_2": ["sim", "proj"],
        "tversky_proj":   ["proj"],
        "no_sirl_ts": ["sim_only"],
    }.get(model_name)
    if banks_present is None:
        raise ValueError(
            f"eval_queries: unrecognized model '{model_name}' "
            f"(expected tversky_sirl, tversky_sirl_2, or tversky_proj)"
        )

    def bank_feature_and_instances(bank_label):
        """(feature_bank (F, D), instance_vectors (N, D)) for one Tversky layer.

        Mirrors compute_layer_quantities in make_tversky_report.py:
          proj — bank = encoder[0].feature_bank; instances = centered *raw* trajectories.
          sim  — bank = tversky_sim.feature_bank; instances = centered model embeddings
                 (model(trajs) returns embeddings; the sim layer is applied separately,
                 which is why it never fires during a forward pass).
        
        fb = feature bank
        iv = instance / input vector (what are tversky features features *of*?)
        """
        if bank_label == "proj":
            fb = model.encoder[0].feature_bank.weight.detach()      # (F, D)
            iv = (trajs_t - trajs_t.mean(0)).detach()               # (N, D)
        elif bank_label == "sim":
            fb = model.tversky_sim.feature_bank.weight.detach()     # (F, D)
            with torch.no_grad():
                embeds = model(trajs_t)                             # model embeddings
            iv = (embeds - embeds.mean(0)).detach()                 # (N, D)
        elif bank_label == "sim_only":
            fb = model.tversky_sim.feature_bank.weight.detach()     # (F, D)
            iv = (trajs_t - trajs_t.mean(0)).detach()               # (N, D)

        else:
            raise ValueError(f"unknown bank label {bank_label!r}")

        if iv.shape[1] != fb.shape[1]:
            raise ValueError(
                f"'{bank_label}' bank: instance vectors {tuple(iv.shape)} and feature "
                f"bank {tuple(fb.shape)} don't share a dim."
            )
        return fb, iv

    def eval_one_bank(feature_bank, instance_vectors):
        def run_query(a_idx, b_idx):
            """s(a) - s(b): retrieve instances salient for a's features but not b's."""
            return retrieve_semantic_expression(
                instance_vectors=instance_vectors,
                feature_bank=feature_bank,
                expression=f"s({a_idx})-s({b_idx})",
                top_feature_count=top_feature_count,
                top_result_count=top_result_count,
            )

        bank_results = {}
        for f_ix, f_name in enumerate(FEATURE_NAMES):
            vals = all_feats[:, f_ix]
            # every traj tied at the absolute max / min feature value (e.g. laptop == 1 / 0)
            max_set = [int(i) for i in np.flatnonzero(vals == vals.max())]
            min_set = [int(i) for i in np.flatnonzero(vals == vals.min())]
            pairs = list(itertools.product(max_set, min_set))    # all (max_traj, min_traj) pairs
            if max_pairs is not None and len(pairs) > max_pairs:
                sample_ix = rng.choice(len(pairs), size=max_pairs, replace=False)
                pairs = [pairs[i] for i in sample_ix]

            # counters aggregated over all pairs for this feature
            n_nonempty_hist = {0: 0, 1: 0, 2: 0}   # how many of the 2 queries per pair were non-empty
            n_both_nonempty = 0                    # pairs where both queries were non-empty (t-test eligible)
            n_ttest_run     = 0                    # pairs where the one-sided t-test actually ran
            n_sig           = {a: 0 for a in alphas}
            pair_records    = []

            for m, n in pairs:
                # 2 queries: max-min (expect higher feature values) and min-max
                res_maxmin = run_query(m, n)   # s(max) - s(min)
                res_minmax = run_query(n, m)   # s(min) - s(max)
                maxmin_ok = res_maxmin["feature_count"] > 0
                minmax_ok = res_minmax["feature_count"] > 0
                n_nonempty = int(maxmin_ok) + int(minmax_ok)
                n_nonempty_hist[n_nonempty] += 1

                rec = {"pair": (m, n), "n_nonempty": n_nonempty} if save_pairs else None

                if n_nonempty == 2:
                    n_both_nonempty += 1
                    # raw feature values of the instances each query retrieved
                    a = [all_feats[inst["item_ix"], f_ix] for inst in res_maxmin["top_instances"]]
                    b = [all_feats[inst["item_ix"], f_ix] for inst in res_minmax["top_instances"]]
                    # one-sided: max-min set should have the LARGER mean feature value.
                    # skip pairs where both retrieved sets are constant: the t-test is
                    # undefined (nan) and scipy warns on catastrophic cancellation.
                    both_constant = np.std(a) == 0 and np.std(b) == 0
                    if len(a) >= 2 and len(b) >= 2 and not both_constant:
                        # near-constant (but not exactly equal) samples make scipy warn
                        # about precision loss; the resulting t is still finite/usable.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            t = stats.ttest_ind(a, b, alternative="greater")
                        n_ttest_run += 1
                        for a_ in alphas:
                            if t.pvalue < a_:
                                n_sig[a_] += 1
                        if save_pairs:
                            rec |= {
                                "mean_maxmin": float(np.mean(a)),
                                "mean_minmax": float(np.mean(b)),
                                "t": float(t.statistic),
                                "p": float(t.pvalue),
                                "sig": {a_: bool(t.pvalue < a_) for a_ in alphas},
                            }
                if save_pairs:
                    pair_records.append(rec)

            feat_result = {
                "n_pairs": len(pairs),
                "n_max_trajs": len(max_set),
                "n_min_trajs": len(min_set),
                "n_nonempty_hist": n_nonempty_hist,       # {0,1,2 -> count of pairs}
                "n_both_nonempty": n_both_nonempty,
                "n_ttest_run": n_ttest_run,
                "n_significant": n_sig,                    # {alpha -> count of significant pairs}
                "frac_significant": {a: (n_sig[a] / n_ttest_run if n_ttest_run else 0.0)
                                     for a in alphas},
            }
            if save_pairs:
                feat_result["pairs"] = pair_records
            bank_results[f_name] = feat_result

        return bank_results

    results = {}
    for bank_label in banks_present:
        fb, iv = bank_feature_and_instances(bank_label)
        results[bank_label] = eval_one_bank(fb, iv)

    return results


def tversky_sim_tsne(config, data, model):
    expt_name = config["experiment_name"]

    all_trajs = data["trajs"]
    all_feats = data["features"]

    # TODO add encoder
    all_sim = model.similarity(torch.tensor(all_trajs, dtype=torch.float32), torch.tensor(all_trajs, dtype=torch.float32))
    t_sne = TSNE(
        n_components=2,
        init="random",
        random_state=0,
        metric="precomputed" # we pass in pairways similarity matrix
    )
    ts_t_sne = t_sne.fit_transform(all_sim.detach().numpy())

    results_dir = Path("results") / expt_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for feat_idx, feat in enumerate(["laptop", "upright"]):
        fig = px.scatter(x=ts_t_sne[:,0], 
                        y=ts_t_sne[:,1],
                        color= all_feats[:, feat_idx],
                        title=f"{expt_name} similarity, colored by {feat}"
                        )
        fig.write_image(results_dir / f"{expt_name}_tsne_feat_{feat}.png")
    return results_dir