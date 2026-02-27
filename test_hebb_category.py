import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from networks import HebbFeatureLayer


# ============================================================
# Load Embeddings
# ============================================================

def load_experiment(experiment_dir):
    experiment_dir = Path(experiment_dir)
    class_dirs = sorted([p for p in experiment_dir.iterdir() if p.is_dir()])
    if len(class_dirs) == 0:
        raise ValueError(f"No category subfolders found in {experiment_dir}")

    x_list = []
    y_list = []
    class_names = []
    class_counts = {}

    for class_idx, class_dir in enumerate(class_dirs):
        class_names.append(class_dir.name)
        npy_files = sorted(class_dir.glob("*.npy"))
        if len(npy_files) == 0:
            continue

        for npy_file in npy_files:
            vec = np.load(npy_file)
            vec = torch.as_tensor(vec, dtype=torch.float32).reshape(-1)
            x_list.append(vec)
            y_list.append(class_idx)

        class_counts[class_dir.name] = len(npy_files)

    if len(x_list) == 0:
        raise ValueError(f"No .npy files found under {experiment_dir}")

    x = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)

    if x.shape[1] != 512:
        raise ValueError(f"Expected 512-d embeddings, got {tuple(x.shape)}")

    return x, y, class_names, class_counts


# ============================================================
# Train/Test Split
# ============================================================

def stratified_split(y, test_frac=0.2, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)

    train_idx = []
    test_idx = []

    for c in torch.unique(y):
        class_idx = torch.where(y == c)[0]
        perm = class_idx[torch.randperm(len(class_idx), generator=g)]
        n_test = max(1, int(round(len(class_idx) * test_frac)))
        n_test = min(n_test, len(class_idx) - 1) if len(class_idx) > 1 else 1
        test_idx.append(perm[:n_test])
        train_idx.append(perm[n_test:])

    return torch.cat(train_idx), torch.cat(test_idx)


def standardize(train_x, test_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


# ============================================================
# Linear Probe
# ============================================================

def train_logistic_classifier(
    train_x, train_y, test_x, test_y,
    num_classes, lr=1e-2, weight_decay=1e-4,
    epochs=400, device="cpu"
):
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    model = nn.Linear(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(train_x).argmax(1) == train_y).float().mean().item()
        test_acc = (model(test_x).argmax(1) == test_y).float().mean().item()

    return {"train_acc": train_acc, "test_acc": test_acc}


# ============================================================
# Load Hebb Model
# ============================================================

def load_hebb_model(checkpoint_path, device="cpu"):
    state = torch.load(checkpoint_path, map_location="cpu")

    Nx = 512
    Nh = 32
    Ny = 1
    d = 128

    model = HebbFeatureLayer(init=[d, Nh, Ny], Nx=Nx)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


# ============================================================
# Representation Extractors
# ============================================================

def extract_hebb_embedding(model, x, device="cpu"):
    """
    Extract compressed embedding (128-d feature layer output).
    """
    with torch.no_grad():
        return model.featurizer(x.to(device)).cpu()


def extract_hidden_states(model, x, device="cpu"):
    """
    Extract hidden activations h (32-d) WITHOUT updating Hebbian matrix A.
    """
    hidden_list = []
    model.reset_state()

    # --- Disable plasticity ---
    original_plastic = model.plastic.clone()
    model.plastic = torch.tensor(False)

    with torch.no_grad():
        for i in range(x.shape[0]):
            x_i = x[i].to(device)
            _, h, _, _ = model(x_i, isFam=False, debug=True) 
			#isFam=false is a dummy var, since update_hebb only updates A when both isFam=false AND self.plastic=True.
            hidden_list.append(h.cpu())

    # --- Restore original plasticity setting ---
    model.plastic = original_plastic

    return torch.stack(hidden_list)


# ============================================================
# Evaluate One Experiment
# ============================================================

def evaluate_experiment(exp_dir, hebb_model, args):
    x, y, class_names, class_counts = load_experiment(exp_dir)
    train_idx, test_idx = stratified_split(y, args.test_frac, args.seed)

    train_y = y[train_idx]
    test_y = y[test_idx]

    # ---------------- Raw 512-d ----------------
    raw_train_x, raw_test_x = standardize(x[train_idx], x[test_idx])
    raw_res = train_logistic_classifier(
        raw_train_x, train_y,
        raw_test_x, test_y,
        len(class_names),
        args.lr, args.weight_decay, args.epochs, args.device
    )

    # ---------------- Hebb 128-d Embedding ----------------
    embed_x = extract_hebb_embedding(hebb_model, x, args.device)
    embed_train_x, embed_test_x = standardize(
        embed_x[train_idx], embed_x[test_idx]
    )
    embed_res = train_logistic_classifier(
        embed_train_x, train_y,
        embed_test_x, test_y,
        len(class_names),
        args.lr, args.weight_decay, args.epochs, args.device
    )

    # ---------------- Hidden h (32-d) ----------------
    hidden_x = extract_hidden_states(hebb_model, x, args.device)
    hidden_train_x, hidden_test_x = standardize(
        hidden_x[train_idx], hidden_x[test_idx]
    )
    hidden_res = train_logistic_classifier(
        hidden_train_x, train_y,
        hidden_test_x, test_y,
        len(class_names),
        args.lr, args.weight_decay, args.epochs, args.device
    )

    return {
        "n_samples": int(x.shape[0]),
        "n_classes": int(len(class_names)),
        "raw_512d": raw_res,
        "hebb_embedding_128d": embed_res,
        "hidden_h_32d": hidden_res,
    }


# ============================================================
# Plot
# ============================================================

def save_bar_plot(results, save_path):
    experiments = sorted(results["experiments"].keys())

    raw_acc = [results["experiments"][e]["raw_512d"]["test_acc"] for e in experiments]
    embed_acc = [results["experiments"][e]["hebb_embedding_128d"]["test_acc"] for e in experiments]
    hidden_acc = [results["experiments"][e]["hidden_h_32d"]["test_acc"] for e in experiments]

    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width, raw_acc, width, label="Raw 512-d")
    ax.bar(x, embed_acc, width, label="Hebb Embedding (128-d)")
    ax.bar(x + width, hidden_acc, width, label="Hidden h (32-d)")

    ax.set_ylabel("Test Accuracy")
    ax.set_title("Category Classification Accuracy by Representation")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-root", default="Embeddings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/hebb_category_comparison.json")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hebb_model = load_hebb_model(args.checkpoint, args.device)

    root = Path(args.embeddings_root)
    experiments = sorted([p for p in root.iterdir() if p.is_dir()])

    results = {"experiments": {}}

    for exp_dir in experiments:
        exp_res = evaluate_experiment(exp_dir, hebb_model, args)
        results["experiments"][exp_dir.name] = exp_res

        print(f"\n[{exp_dir.name}]")
        print(
            f"raw={exp_res['raw_512d']['test_acc']:.4f} | "
            f"embed={exp_res['hebb_embedding_128d']['test_acc']:.4f} | "
            f"hidden={exp_res['hidden_h_32d']['test_acc']:.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_path = out_path.with_name(out_path.stem + "_barplot.png")
    save_bar_plot(results, plot_path)

    print("\nSaved results to:", out_path)
    print("Saved plot to:", plot_path)


if __name__ == "__main__":
    main()