import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


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
		raise ValueError(f"Expected 512-d embeddings, got {tuple(x.shape)} for {experiment_dir}")

	return x, y, class_names, class_counts


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

	train_idx = torch.cat(train_idx)
	test_idx = torch.cat(test_idx)
	return train_idx, test_idx


def standardize(train_x, test_x):
	mean = train_x.mean(dim=0, keepdim=True)
	std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
	return (train_x - mean) / std, (test_x - mean) / std


def train_logistic_classifier(
	train_x,
	train_y,
	test_x,
	test_y,
	num_classes,
	lr=1e-2,
	weight_decay=1e-4,
	epochs=400,
	device="cpu",
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
		logits = model(train_x)
		loss = criterion(logits, train_y)
		loss.backward()
		opt.step()

	model.eval()
	with torch.no_grad():
		train_pred = model(train_x).argmax(dim=1)
		test_pred = model(test_x).argmax(dim=1)
		train_acc = (train_pred == train_y).float().mean().item()
		test_acc = (test_pred == test_y).float().mean().item()

	return {
		"train_acc": train_acc,
		"test_acc": test_acc,
	}


def load_featurizer_from_checkpoint(checkpoint_path, device="cpu"):
	state = torch.load(checkpoint_path, map_location="cpu")
	if "featurizer.weight" not in state or "featurizer.bias" not in state:
		raise ValueError("Checkpoint missing featurizer weights")

	out_dim, in_dim = state["featurizer.weight"].shape
	featurizer = nn.Linear(in_dim, out_dim)
	featurizer.weight.data.copy_(state["featurizer.weight"].float())
	featurizer.bias.data.copy_(state["featurizer.bias"].float())
	featurizer.to(device)
	featurizer.eval()
	return featurizer


def apply_featurizer(featurizer, x, device="cpu"):
	with torch.no_grad():
		return featurizer(x.to(device)).cpu()


def evaluate_experiment(exp_dir, featurizer, args):
	x, y, class_names, class_counts = load_experiment(exp_dir)
	train_idx, test_idx = stratified_split(y, test_frac=args.test_frac, seed=args.seed)

	raw_train_x = x[train_idx]
	raw_test_x = x[test_idx]
	train_y = y[train_idx]
	test_y = y[test_idx]

	raw_train_x, raw_test_x = standardize(raw_train_x, raw_test_x)
	raw_res = train_logistic_classifier(
		raw_train_x,
		train_y,
		raw_test_x,
		test_y,
		num_classes=len(class_names),
		lr=args.lr,
		weight_decay=args.weight_decay,
		epochs=args.epochs,
		device=args.device,
	)

	feat_x = apply_featurizer(featurizer, x, device=args.device)
	feat_train_x = feat_x[train_idx]
	feat_test_x = feat_x[test_idx]
	feat_train_x, feat_test_x = standardize(feat_train_x, feat_test_x)
	feat_res = train_logistic_classifier(
		feat_train_x,
		train_y,
		feat_test_x,
		test_y,
		num_classes=len(class_names),
		lr=args.lr,
		weight_decay=args.weight_decay,
		epochs=args.epochs,
		device=args.device,
	)

	return {
		"n_samples": int(x.shape[0]),
		"n_classes": int(len(class_names)),
		"class_names": class_names,
		"class_counts": class_counts,
		"split": {
			"train": int(len(train_idx)),
			"test": int(len(test_idx)),
			"test_frac": float(args.test_frac),
		},
		"raw_512d": raw_res,
		"hebb_featurized_50d": feat_res,
		"test_acc_delta": feat_res["test_acc"] - raw_res["test_acc"],
	}


def save_bar_plot(results, save_path):
	experiments = sorted(results["experiments"].keys())
	raw_acc = [results["experiments"][e]["raw_512d"]["test_acc"] for e in experiments]
	feat_acc = [results["experiments"][e]["hebb_featurized_50d"]["test_acc"] for e in experiments]

	x = np.arange(len(experiments))
	width = 0.38

	fig, ax = plt.subplots(figsize=(9, 5))
	ax.bar(x - width / 2, raw_acc, width, label="Raw 512-d")
	ax.bar(x + width / 2, feat_acc, width, label="Featurized 128-d")

	ax.set_ylabel("Test Accuracy")
	ax.set_title("Category Classification Accuracy by Experiment")
	ax.set_xticks(x)
	ax.set_xticklabels(experiments, rotation=25, ha="right")
	ax.set_ylim(0.0, 1.0)
	ax.legend()
	ax.grid(axis="y", linestyle=":", alpha=0.5)

	fig.tight_layout()
	fig.savefig(save_path, dpi=150)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(
		description="Compare category separability: raw 512-d embeddings vs HebbFF featurized embeddings"
	)
	parser.add_argument("--embeddings-root", default="Embeddings")
	parser.add_argument(
		"--checkpoint",
		default="results/tensor_board_logs_no_freeze_augment_brady/results/HebbFeature_no_freeze_augment_brady_(10).pkl",
	)
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

	featurizer = load_featurizer_from_checkpoint(args.checkpoint, device=args.device)

	root = Path(args.embeddings_root)
	experiments = sorted([p for p in root.iterdir() if p.is_dir()])
	if len(experiments) == 0:
		raise ValueError(f"No experiments found in {root}")

	results = {
		"checkpoint": args.checkpoint,
		"embeddings_root": args.embeddings_root,
		"device": args.device,
		"settings": {
			"test_frac": args.test_frac,
			"epochs": args.epochs,
			"lr": args.lr,
			"weight_decay": args.weight_decay,
			"seed": args.seed,
		},
		"experiments": {},
	}

	for exp_dir in experiments:
		exp_res = evaluate_experiment(exp_dir, featurizer, args)
		results["experiments"][exp_dir.name] = exp_res

		print(f"\n[{exp_dir.name}] n={exp_res['n_samples']} classes={exp_res['n_classes']}")
		print(
			f"raw test_acc={exp_res['raw_512d']['test_acc']:.4f} | "
			f"featurized test_acc={exp_res['hebb_featurized_50d']['test_acc']:.4f} | "
			f"delta={exp_res['test_acc_delta']:+.4f}"
		)

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	plot_path = out_path.with_name(out_path.stem + "_barplot.png")
	save_bar_plot(results, plot_path)

	print(f"\nSaved comparison to: {out_path}")
	print(f"Saved bar plot to: {plot_path}")


if __name__ == "__main__":
	main()
