"""
make_tversky_report.py

Generate a markdown report of Tversky-SIRL feature diagnostics for a trained model.
Based on tversky_features.ipynb.

Usage:
    python make_tversky_report.py \
        --config ../configs/tversky_sirl.yaml \
        --model ../results/tversky_sirl_gridrobot/tversky_sirl_dim10_fbank4_seed0.pth \
        [--out reports/] [--seed 0]

Each report section is its own helper function with signature
    section_fn(ctx: ReportContext) -> list[str]   # returns markdown lines
Add new evals by writing a new section_fn and appending it to SECTIONS.
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless: save figs instead of showing them
import matplotlib.pyplot as plt

from models import load_model_only
from utils import *  # parse_config, load_data

# gridrobot env for visualization
import sys
sys.path.insert(0, "../../../simulated_data/001-gridrobot/")
from gridrobot import Gridrobot

# -----------------------------------------------------------------------------
# env + salience helpers (from notebook)
# -----------------------------------------------------------------------------

GRIDROBOT_CONFIG = {
    "X": 5,
    "Y": 5,
    "obstacles": [],
    "starts": [[0, 0], [0, 4], [4, 0], [4, 4]],
    "goals": [[4, 4], [4, 0], [0, 4], [0, 0]],
    "features": ["computer_dist", "joint_up"],
}

FEATURE_NAMES = ["laptop (computer_dist)", "upright (joint_up)"]


def build_env():
    c = GRIDROBOT_CONFIG
    return Gridrobot(c["X"], c["Y"], c["obstacles"], c["starts"], c["goals"])


def compute_salience(x, feature_bank):
    """
    x is (N, D), feature_bank is (F, D)
    """
    feature_measures = x @ feature_bank.T  # (N, F)
    return F.relu(feature_measures).sum(-1)


def retrieve_semantic_expression(
    instance_vectors: torch.Tensor,  # (N, D)
    feature_bank: torch.Tensor,      # (F, D)
    expression: str,
    top_feature_count: int,
    top_result_count: int,
) -> dict:
    """
    Evaluate a set expression over instance vectors and feature bank.
    Expression uses s(i) notation where i is a dataset item_id (row index).
    Example: "s(0) - s(1)"  ->  features of item 0 minus features of item 1
    (from tversky-networks-iclr2026 semantic_utils.py, prints removed)
    """
    query_item_ixes = []

    def s(item_ix: int) -> set:
        feature_values = instance_vectors[item_ix:item_ix + 1] @ feature_bank.T  # (1, F)
        feature_ixes = []
        for feature_ix in torch.argsort(feature_values[0], descending=True)[:top_feature_count]:
            if feature_values[0][feature_ix] > 0:
                feature_ixes.append(int(feature_ix))
            else:
                break
        query_item_ixes.append(item_ix)
        return set(feature_ixes)

    semantic_features = eval(expression)  # noqa: S307 - internal expressions only

    if not semantic_features:
        return {
            "expression": expression.strip(),
            "query_item_ixes": query_item_ixes,
            "feature_count": 0,
            "top_instances": [],
        }

    semantic_f_bank = torch.index_select(
        feature_bank, 0, torch.tensor(sorted(semantic_features))
    )
    dot = instance_vectors @ semantic_f_bank.T   # (N, |features|)
    p_saliences = F.relu(dot).sum(dim=1)         # (N,)
    p_measures = dot.sum(dim=1)                  # (N,)

    top_instances = []
    for result_ix in torch.argsort(p_measures, descending=True)[:top_result_count]:
        top_instances.append({
            "item_ix": result_ix.item(),
            "salience": p_saliences[result_ix].item(),
            "measure": p_measures[result_ix].item(),
        })
    return {
        "expression": expression.strip(),
        "query_item_ixes": query_item_ixes,
        "feature_count": len(semantic_features),
        "top_instances": top_instances,
    }


# -----------------------------------------------------------------------------
# report context: everything a section might need, computed once
# -----------------------------------------------------------------------------

@dataclass
class ReportContext:
    config: dict
    model: object
    env: object
    all_trajs: np.ndarray          # (N, T)
    all_feats: np.ndarray          # (N, 2)
    feature_bank: torch.Tensor     # (F, D)
    centered_embeds: torch.Tensor  # (N, D)
    centered_salience: torch.Tensor  # (N,)
    all_sim: torch.Tensor          # (N, N) pairwise Tversky similarity
    fig_dir: Path                  # where to save trajectory figures
    fig_relpath: str               # relative path used inside the markdown
    # cross-section state: semantic-difference results reused by the t-test
    semantic_results: dict = field(default_factory=dict)


def build_context(config_path: str, model_path: str, out_dir: Path) -> ReportContext:
    config = parse_config(config_path)
    model = load_model_only(config, model_path)
    feature_bank = model.tversky_sim.feature_bank.weight.detach()

    data = load_data(config)
    all_trajs = data["trajs"]
    all_feats = data["features"]

    all_embeds = model(torch.as_tensor(all_trajs, dtype=torch.float32))
    # raw saliences are all 0 (features all positive) -> center embeddings
    centered_embeds = (all_embeds - all_embeds.mean(0)).detach()
    centered_salience = compute_salience(centered_embeds, feature_bank).detach()
    all_sim = model.tversky_sim(centered_embeds, centered_embeds).detach()

    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    return ReportContext(
        config=config,
        model=model,
        env=build_env(),
        all_trajs=all_trajs,
        all_feats=all_feats,
        feature_bank=feature_bank,
        centered_embeds=centered_embeds,
        centered_salience=centered_salience,
        all_sim=all_sim,
        fig_dir=fig_dir,
        fig_relpath="figs",
    )


# -----------------------------------------------------------------------------
# figure helper
# -----------------------------------------------------------------------------

def save_traj_fig(ctx: ReportContext, traj_idx: int, tag: str) -> str:
    """
    Visualize one trajectory and save it as a png; returns a markdown image line.
    NOTE: assumes env.visualize_one_traj draws on the current matplotlib figure.
    If it creates its own figure/shows it, adapt here (this is the only place
    figure saving happens).
    """
    fname = f"{tag}_traj{traj_idx}.png"
    fpath = ctx.fig_dir / fname
    plt.figure()
    ctx.env.visualize_one_traj(ctx.all_trajs[traj_idx])
    plt.savefig(fpath, bbox_inches="tight", dpi=120)
    plt.close("all")
    return f"![traj {traj_idx}]({ctx.fig_relpath}/{fname})"


def traj_summary_line(ctx: ReportContext, traj_idx: int) -> str:
    feats = ctx.all_feats[traj_idx]
    sal = ctx.centered_salience[traj_idx].item()
    return (f"traj `{traj_idx}` — features: "
            f"{FEATURE_NAMES[0]} = {feats[0]:.4f}, {FEATURE_NAMES[1]} = {feats[1]:.4f}, "
            f"salience = {sal:.4f}")


# -----------------------------------------------------------------------------
# report sections
# each takes ctx, returns list of markdown lines
# -----------------------------------------------------------------------------

def section_similar_dissimilar(ctx: ReportContext, n_queries: int = 2, top_n: int = 3) -> list:
    """Random sample query trajectories; show their most / least similar."""
    lines = ["# sample similar + dissimilar trajectories", ""]
    sim = ctx.all_sim.numpy()
    query_idxs = random.sample(range(len(ctx.all_trajs)), n_queries)

    for q in query_idxs:
        order = np.argsort(sim[q])          # ascending
        least = order[:top_n]
        most = [i for i in order[::-1] if i != q][:top_n]  # skip self-match

        lines += [f"## query {traj_summary_line(ctx, q)}", "",
                  save_traj_fig(ctx, q, "query"), "",
                  f"### {top_n} most similar", ""]
        for i in most:
            lines += [f"- {traj_summary_line(ctx, i)} (sim = {sim[q, i]:.4f})",
                      "", save_traj_fig(ctx, i, f"q{q}_most"), ""]
        lines += [f"### {top_n} least similar", ""]
        for i in least:
            lines += [f"- {traj_summary_line(ctx, i)} (sim = {sim[q, i]:.4f})",
                      "", save_traj_fig(ctx, i, f"q{q}_least"), ""]
    return lines


def section_salience_sorted(ctx: ReportContext, sample_n: int = 10) -> list:
    """Sample trajectories evenly from min to max salience."""
    lines = ["# sort trajectories by salience", "",
             f"{sample_n} trajectories sampled evenly from least to most salient.", ""]
    sorted_idxs = ctx.centered_salience.argsort()
    step = len(sorted_idxs) // sample_n
    sampled = sorted_idxs[[i for i in range(0, len(sorted_idxs), step)][:sample_n]]

    for rank, i in enumerate(sampled):
        i = int(i)
        lines += [f"## salience rank {rank + 1}/{sample_n}: {traj_summary_line(ctx, i)}",
                  "", save_traj_fig(ctx, i, f"salience_rank{rank + 1}"), ""]
    return lines


def section_semantic_differences(ctx: ReportContext,
                                 top_feature_count: int = 4,
                                 top_result_count: int = 10) -> list:
    """For each raw feature: (max - min) and (min - max) semantic set expressions.
    Stores results in ctx.semantic_results for downstream sections (t-test)."""
    lines = ["# max feature value - min feature value t-test of means", ""]

    for f_ix, f_name in enumerate(FEATURE_NAMES):
        min_idx = int(np.argmin(ctx.all_feats[:, f_ix]))
        max_idx = int(np.argmax(ctx.all_feats[:, f_ix]))

        for direction, (a, b) in [("max - min", (max_idx, min_idx)),
                                  ("min - max", (min_idx, max_idx))]:
            res = retrieve_semantic_expression(
                instance_vectors=ctx.centered_embeds,
                feature_bank=ctx.feature_bank,
                expression=f"s({a})-s({b})",
                top_feature_count=top_feature_count,
                top_result_count=top_result_count,
            )
            ctx.semantic_results[(f_ix, direction)] = res

            lines += [f"## {f_name}: {direction}", "",
                      f"expression `{res['expression']}` → "
                      f"{res['feature_count']} Tversky feature(s) in the difference set", ""]
            lines += [f"**{traj_summary_line(ctx, a)}**", "",
                      save_traj_fig(ctx, a, f"feat{f_ix}_{direction.replace(' ', '')}_a"), "",
                      "minus", "",
                      f"**{traj_summary_line(ctx, b)}**", "",
                      save_traj_fig(ctx, b, f"feat{f_ix}_{direction.replace(' ', '')}_b"), "",
                      "equals top instances:", ""]
            if not res["top_instances"]:
                lines += ["*(empty feature set — no instances retrieved)*", ""]
            for inst in res["top_instances"]:
                i = inst["item_ix"]
                lines += [f"- {traj_summary_line(ctx, i)} "
                          f"(measure = {inst['measure']:.4f}, salience = {inst['salience']:.4f})",
                          "", save_traj_fig(ctx, i, f"feat{f_ix}_{direction.replace(' ', '')}_top"), ""]
    return lines


def section_ttest(ctx: ReportContext) -> list:
    """If both (max-min) and (min-max) sets are non-empty for a feature, run a
    t-test of means of the raw feature values over the retrieved instances."""
    lines = ["## t-test of means", ""]
    if not ctx.semantic_results:
        return lines + ["*(semantic-differences section did not run before this one)*", ""]

    any_ran = False
    for f_ix, f_name in enumerate(FEATURE_NAMES):
        max_min = ctx.semantic_results.get((f_ix, "max - min"), {})
        min_max = ctx.semantic_results.get((f_ix, "min - max"), {})
        a = [ctx.all_feats[inst["item_ix"], f_ix] for inst in max_min.get("top_instances", [])]
        b = [ctx.all_feats[inst["item_ix"], f_ix] for inst in min_max.get("top_instances", [])]

        if not a or not b:
            lines += [f"- **{f_name}**: skipped — at least one feature set is empty "
                      f"(max-min: {len(a)} instances, min-max: {len(b)} instances)", ""]
            continue

        any_ran = True
        t = stats.ttest_ind(a, b)
        lines += [f"- **{f_name}**: mean(max-min set) = {np.mean(a):.4f}, "
                  f"mean(min-max set) = {np.mean(b):.4f}, "
                  f"t = {t.statistic:.4f}, p = {t.pvalue:.4g}", ""]

    if not any_ran:
        lines += ["*Can't run any t-tests on this model — the feature sets are empty.*", ""]
    return lines


# to add a new eval: write section_myeval(ctx) -> list[str] and append here.
# sections run in order; ctx.semantic_results lets later sections use earlier results.
SECTIONS = [
    section_similar_dissimilar,
    section_salience_sorted,
    section_semantic_differences,
    section_ttest,
]


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to yaml config")
    parser.add_argument("--model", required=True, help="path to model checkpoint (.pth)")
    parser.add_argument("--out", default="reports", help="output directory")
    parser.add_argument("--seed", type=int, default=None, help="random seed for query sampling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = Path(args.out) / Path(args.model).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = build_context(args.config, args.model, out_dir)

    lines = [f"# Tversky feature report: `{Path(args.model).name}`", "",
             f"config: `{args.config}`  ",
             f"trajectories: {ctx.all_trajs.shape}, features: {ctx.all_feats.shape}, "
             f"feature bank: {tuple(ctx.feature_bank.shape)}", ""]
    for section_fn in SECTIONS:
        print(f"running {section_fn.__name__}...")
        lines += section_fn(ctx)
        lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()