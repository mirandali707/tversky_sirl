from pathlib import Path

import pandas as pd
import plotly.express as px

# directories are relative to the experiment root (parent of src/)
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGS_DIR = EXPERIMENT_DIR / "figs"

# map result subdir name -> base method name
METHOD_DIRS = {
    "random_gridrobot": "random",
    "pca_gridrobot": "pca",
    "sirl_gridrobot": "sirl",
    "tversky_sirl_gridrobot": "tversky_sirl",
    "tversky_sirl_2_gridrobot": "tversky_sirl_2",
}


def load_all_results():
    """Aggregate every results.csv under RESULTS_DIR into one df.

    Adds a `method` column. For the tversky methods each `fbank_size` is
    treated as its own method (e.g. "tversky_sirl (fbank=8)").
    """
    frames = []
    for subdir, base_name in METHOD_DIRS.items():
        csv_path = RESULTS_DIR / subdir / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["method"] = [_method_label(base_name, row) for _, row in df.iterrows()]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _method_label(base_name, row):
    """Name a method, folding fbank_size into the name when present."""
    fbank = row.get("fbank_size")
    if pd.notna(fbank):
        return f"{base_name} (fbank={int(fbank)})"
    return base_name


def average_over_seeds(df):
    """Mean fpe and tpa per (method, latent_dim), averaging over seeds.

    `tpa_mean` is the per-checkpoint mean over thetas; we average that over
    seeds to get a single tpa value per method/dim.
    """
    return (
        df.groupby(["method", "latent_dim"], as_index=False)
        .agg(fpe=("fpe", "mean"), tpa=("tpa_mean", "mean"))
    )


def plot_metric(df_dim, dim, metric, title_label):
    """Bar chart of `metric` for each method at a fixed latent dim."""
    df_sorted = df_dim.sort_values(metric, ascending=False)
    fig = px.bar(
        df_sorted,
        x="method",
        y=metric,
        color="method",
        title=f"gridrobot {title_label} for representation dim {dim}",
    )
    fig.update_layout(
        xaxis_title="method",
        yaxis_title=title_label,
        showlegend=False,
    )
    return fig


def save_fig(fig, dim, metric):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGS_DIR / f"gridrobot_{metric}_dim{dim}.png"
    fig.write_image(str(out_path))
    return out_path


def main():
    """
    ```
    seeds: [0, 1, 2]
    latent_dim: [2, 4, 6, 8, 10]
    fbank_size: [2, 4, 8, 16] # for tversky_sirl, tversky_sirl_2 only
    ```
    TODO
    aggregate all results (`results.csv` in each subdir of `results`) into one df
    for each latent_dim:
        get fpe, tpa list from random, pca, sirl, tversky_sirl, tversky_sirl_2
            (for tversky methods, treat each `fbank_size` as its own method)
        average fpe over seeds, plot fpe for each method
        plot average tpa for each method
        save plot for this latent dim (include latent dim in title, e.g. f"gridrobot {fpe/tpa} for representation dim {dim}")
        in figs/
    """
    df = load_all_results()
    averaged = average_over_seeds(df)

    for dim in sorted(averaged["latent_dim"].unique()):
        df_dim = averaged[averaged["latent_dim"] == dim]
        for metric in ("fpe", "tpa"):
            fig = plot_metric(df_dim, dim, metric, metric)
            out_path = save_fig(fig, dim, metric)
            print(f"saved {out_path}")


if __name__ == '__main__':
    main()
