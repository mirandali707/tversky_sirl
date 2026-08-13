"""Stacked bars: nonempty-histogram + within-bucket significance, per feature.

Each bar stacks (bottom -> top):
  neither nonempty : n_nonempty_hist[0]
  one nonempty     : n_nonempty_hist[1]
  then n_nonempty_hist[2] (== what the t-test ran on) split by NESTED significance
  into incremental bands: both/n.s. | p<0.05 | p<0.01 | p<0.001  (sum == hist[2])
Counts summed across seeds (and any other non-x dimension).
One figure per sweep column present among X_CANDIDATES; within a figure,
one subplot row per Tversky bank present (query_proj / query_sim), one col per feature.
"""
import ast, sys
import pandas as pd
from plotly.subplots import make_subplots

SEGMENTS = [                       # (legend label, color); bottom -> top
    ("neither nonempty", "#e0e0e0"),
    ("one nonempty",     "#9e9e9e"),
    ("both, n.s.",       "#cfe1f2"),
    ("both, p<0.05",     "#7fb8de"),
    ("both, p<0.01",     "#3a8bc7"),
    ("both, p<0.001",    "#1b4f8a"),
]
LABELS = [s[0] for s in SEGMENTS]
X_CANDIDATES = ["latent_dim", "fbank_size", "decoder_hidden"]  # any present -> its own figure
BANK_ORDER   = ["proj", "sim", "sim_only"]     # query_proj / query_sim -> subplot rows, in this order

def _row_segments(fd):
    h, s = fd["n_nonempty_hist"], fd["n_significant"]
    h0, h1, h2 = h.get(0, 0), h.get(1, 0), h.get(2, 0)
    s05, s01, s001 = s.get(0.05, 0), s.get(0.01, 0), s.get(0.001, 0)
    return [h0, h1, h2 - s05, s05 - s01, s01 - s001, s001]

def _tidy(df):
    # each bank now lives in its own column: query_proj / query_sim (either may be absent
    # or NaN for a given row, since a model only has the banks it exposes).
    bank_cols = [f"query_{b}" for b in BANK_ORDER if f"query_{b}" in df.columns]
    if not bank_cols:
        raise ValueError("no query_* columns found (expected query_proj / query_sim)")
    xcols = [c for c in X_CANDIDATES if c in df.columns]
    feats, rows = None, []
    for col in bank_cols:
        bank = col[len("query_"):]
        for _, r in df.iterrows():
            cell = r[col]
            if isinstance(cell, float) and pd.isna(cell):   # bank absent for this run
                continue
            parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
            if feats is None:
                feats = list(parsed.keys())
            for f in parsed:
                rows.append({"bank": bank, "feature": f,
                             **{c: r[c] for c in xcols},
                             **dict(zip(LABELS, _row_segments(parsed[f])))})
    return pd.DataFrame(rows), feats, xcols

def _figure(tidy, feats, x_col):
    banks = [b for b in BANK_ORDER if b in set(tidy["bank"])]
    fig = make_subplots(rows=len(banks), cols=len(feats), shared_yaxes=True,
                        column_titles=feats, row_titles=banks,
                        horizontal_spacing=0.07, vertical_spacing=0.12)
    first = True                       # legend once, across the whole grid
    for ri, bank in enumerate(banks, 1):
        for ci, f in enumerate(feats, 1):
            sub = tidy[(tidy.bank == bank) & (tidy.feature == f)]
            agg = sub.groupby(x_col, as_index=False)[LABELS].sum().sort_values(x_col)
            x = agg[x_col].astype(str)
            for label, color in SEGMENTS:
                fig.add_bar(x=x, y=agg[label], name=label, marker_color=color,
                            legendgroup=label, showlegend=first, row=ri, col=ci)
            first = False
    fig.update_layout(barmode="stack", template="plotly_white", bargap=0.28,
                      title=f"nonempty + significance by {x_col} (summed over seeds)",
                      legend_title_text="pair category")
    fig.update_xaxes(title_text=x_col, type="category")
    fig.update_yaxes(title_text="# pairs", col=1)
    return fig

def plot_results(csv_path):
    tidy, feats, xcols = _tidy(pd.read_csv(csv_path))
    return {x: _figure(tidy, feats, x) for x in xcols}

if __name__ == "__main__":
    for x, fig in plot_results(sys.argv[1]).items():
        fig.show()