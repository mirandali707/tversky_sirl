import torch

# from tversky-networks-iclr2026/experiments/103-vision-nabirds/src/semantic_utils.py
def retrieve_semantic_expression(
    instance_vectors: torch.Tensor,   # (N, D)  — all instance reps from parquet
    feature_bank: torch.Tensor,        # (F, D)  — from .npy
    expression: str,
    top_feature_count: int,
    top_result_count: int,
) -> dict:
    """
    Evaluate a set expression over instance vectors and feature bank.
    Expression uses s(i) notation where i is a dataset item_id (row index).
    Example: "s(0) - s(1)"  →  features of item 0 minus features of item 1
    """
    query_item_ixes = []

    def s(item_ix: int) -> set:
        print(item_ix)
        feature_values = instance_vectors[item_ix:item_ix+1] @ feature_bank.T  # (1, F)
        print("feature_values")
        print(feature_values)
        feature_ixes = []
        for feature_ix in torch.argsort(feature_values[0], descending=True)[:top_feature_count]:
            print("loop")
            print(feature_values[0][feature_ix])
            if feature_values[0][feature_ix] > 0:
                feature_ixes.append(int(feature_ix))
            else:
                break
        query_item_ixes.append(item_ix)
        print(set(feature_ixes))
        return set(feature_ixes)

    semantic_features = eval(expression)

    if not semantic_features:
        # logging.warning("Expression evaluated to empty feature set.")
        return {"expression": expression, "query_item_ixes": [], "top_instances": [], "feature_count": 0}

    semantic_f_bank = torch.index_select(
        feature_bank, 0, torch.tensor(sorted(semantic_features))
    )
    dot = instance_vectors @ semantic_f_bank.T          # (N, |features|)
    p_saliences = F.relu(dot).sum(dim=1)                # (N,)
    p_measures  = dot.sum(dim=1)                        # (N,)

    top_instances = []
    for result_ix in torch.argsort(p_measures, descending=True)[:top_result_count]:
        top_instances.append({
            "item_ix"  : result_ix.item(),
            "salience" : p_saliences[result_ix].item(),
            "measure"  : p_measures[result_ix].item(),
        })

    return {
        "expression"      : expression.strip(),
        "query_item_ixes" : query_item_ixes,
        "feature_count"   : len(semantic_features),
        "top_instances"   : top_instances,
    }