import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from data import GenRecogClassifyData
from networks import HebbFeatureLayer
from plotting import get_recog_positive_rates


def load_hebb_feature_layer(pretrained_weights, Nx, Ny, Nh, d):
    net = HebbFeatureLayer(init=[d, Nh, Ny], Nx=Nx)
    # Load pretrained weights
    net.load(pretrained_weights)
    return net


def load_experiment_embeddings(experiment_dir):
    experiment_dir = Path(experiment_dir)
    x_list = []
    category_counts = {}

    for npy_path in sorted(experiment_dir.rglob("*.npy")):
        rel = npy_path.relative_to(experiment_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        emb = np.load(npy_path)
        emb = torch.as_tensor(emb, dtype=torch.float32).reshape(-1)
        x_list.append(emb)
        category_counts[category] = category_counts.get(category, 0) + 1

    if len(x_list) == 0:
        raise ValueError(f"No .npy files found in {experiment_dir}")

    x = torch.stack(x_list, dim=0)
    if x.shape[1] != 512:
        raise ValueError(f"Expected 512-d embeddings, got shape {tuple(x.shape)} for {experiment_dir}")
    return x, category_counts


def evaluate_experiment(net, embeddings, device, stop_at_r=None):        
    dummy_classes = torch.zeros(embeddings.shape[0], 1, device=device)
    sample_space = TensorDataset(embeddings.to(device), dummy_classes)
    generator = GenRecogClassifyData(sampleSpace=sample_space)

    t = embeddings.shape[0]
    max_r = max(2, t // 2)
    if stop_at_r is None:
        stop_at_r = max_r
    else:
        stop_at_r = min(stop_at_r, max_r)

    def gen_data(r):
        x, y = generator(t, r, P=0.5, batchSize=None, multiRep=False, device=device).tensors
        return TensorDataset(x, y[..., 0:1])

    test_r, test_acc, true_pos, false_pos = get_recog_positive_rates(
        net,
        gen_data,
        xscale="log",
        upToR=1,
        stopAtR=stop_at_r,
    )

    summary = {
        "n_items": int(t),
        "stop_at_r": int(stop_at_r),
        "n_points": int(len(test_r)),
        "max_r_tested": int(max(test_r)) if len(test_r) else None,
        "best_acc": float(max(test_acc)) if len(test_acc) else None,
        "acc_at_last_r": float(test_acc[-1]) if len(test_acc) else None,
        "curves": {
            "R": [int(r) for r in test_r],
            "acc": [float(a) for a in test_acc],
            "true_pos_rate": [float(v) for v in true_pos],
            "false_pos_rate": [float(v) for v in false_pos],
        },
    }
    return summary


def plot_results(results, save_path):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for exp_name, summary in results["experiments"].items():
        curve = summary["curves"]
        r = curve["R"]
        if len(r) == 0:
            continue
        ax[0].plot(r, curve["acc"], marker="o", label=exp_name)
        ax[1].plot(r, curve["true_pos_rate"], marker="o", linestyle="-", label=f"{exp_name} TP")
        ax[1].plot(r, curve["false_pos_rate"], marker="o", linestyle="--", label=f"{exp_name} FP")

    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("Probability")
    ax[1].set_xlabel("R")
    ax[0].set_title("HebbFeatureLayer on Embeddings")
    ax[0].legend()
    ax[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HebbFeatureLayer checkpoint on Embeddings experiments")
    parser.add_argument(
        "--checkpoint",
        default="results/tensor_board_logs_no_freeze/results/HebbFeature_no_freeze_(6).pkl",
    )
    parser.add_argument("--embeddings-root", default="Embeddings")
    parser.add_argument("--stop-at-r", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/embedding_eval_hebbfeature_no_freeze_6.json")
    args = parser.parse_args()

    net = load_hebb_feature_layer(args.checkpoint, 512, 1, 16, 50)

    root = Path(args.embeddings_root)
    experiments = sorted([p for p in root.iterdir() if p.is_dir()])
    if len(experiments) == 0:
        raise ValueError(f"No experiment folders found under {root}")

    results = {
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "experiments": {},
    }

    for exp_dir in experiments:
        x, category_counts = load_experiment_embeddings(exp_dir)
        summary = evaluate_experiment(net, x, device=args.device, stop_at_r=args.stop_at_r)
        summary["category_counts"] = category_counts
        results["experiments"][exp_dir.name] = summary

        print(f"\n[{exp_dir.name}] n_items={summary['n_items']} stop_at_r={summary['stop_at_r']}")
        print(f"best_acc={summary['best_acc']:.4f} last_acc={summary['acc_at_last_r']:.4f} points={summary['n_points']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    json_out = out_path.with_suffix(".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_out = out_path.with_suffix(".png")
    plot_results(results, plot_out)

    print(f"\nSaved results to: {out_path}")
    print(f"Saved json to: {json_out}")
    print(f"Saved plot to: {plot_out}")


if __name__ == "__main__":
    main()