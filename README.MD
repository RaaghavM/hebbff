# HebbFF-mod: Learned Featurizers, Image Augmentation, and Category Information in a Continual Familiarity-Detection Network

This repo extends the Hebbian feed-forward familiarity-detection network from
Tyulmankov, Yang & Abbott ("Meta-learning synaptic plasticity and memory addressing
for continual familiarity detection", *Neuron* 2022 — the original `HebbFF` model in
`networks.py`/`net_utils.py`/`plotting.py`/`data.py`/`bogacz.py`/`plot_figs.py`) with a
**learned front-end featurizer** that maps ResNet18 image embeddings down to the
network's input dimension, trains it end-to-end with the Hebbian recognition-memory
network, and studies:

1. How well the network generalizes to novel repeat-lags **R** with and without image
   **augmentation** during training.
2. How the **Fisher Linear Discriminant (FLD) readout weights** (`W_FLD`) over the
   hidden layer differ between augmented and non-augmented models.
3. Whether **object category** can still be linearly decoded from the network's
   intermediate (bottlenecked) representation, compared to the raw ResNet embedding.
4. Whether an image's **memorability** (how reliably the network recognizes it as
   familiar) relates to how **decodable its category** is, and whether that
   relationship changes across models (augmented vs. non-augmented, raw ResNet vs.
   Hebb-featurized).

The `archive/` folder contains the original paper's exploratory scripts and is kept
for reference only — it is not part of the reproduction pipeline below.

---

## 1. Background: the base HebbFF model

A `HebbNet` (`networks.py`) is a 2-layer network whose first-layer weight matrix
`W1` is augmented at every timestep by a fast Hebbian trace `A(t)`:

```
A(t+1) = lam * A(t) + eta * h(t) x(t)^T
h(t)   = sigmoid( (W1 + A(t)) x(t) + b1 )
y(t)   = sigmoid( W2 h(t) + b2 )
```

`x(t)` is presented once per timestep; the label `y(t)=1` iff `x(t)` repeats an item
seen `R` steps earlier ("continual familiarity detection" / delayed-recognition task).
`eta`, `lam`, `W1`, `W2`, `b1`, `b2` are all meta-learned by gradient descent through
time on a **curriculum** that starts at a short repeat-lag `R0` and increases `R`
every time the network reaches a target accuracy (`net_utils.train_curriculum`,
`accuracyStopThres`), until it plateaus (`itersToQuit`) or a max lag `Rf` is reached.
`net.hist['increment_R']` records `(iteration, R)` for every curriculum step — this is
what produces the "performance vs. R" curriculum trajectories.

## 2. This project's additions

| Piece | What it does | Where |
|---|---|---|
| **Learned featurizer** | A linear (or linear+ReLU) layer `Nx → d` in front of the Hebbian core, so the network can take raw 512-d ResNet18 features as input instead of hand-designed binary vectors. | `networks.HebbFeatureLayer`, `networks.HebbExtendedFeatureLayer` |
| **Image-recognition data generator** | Builds "recognize the repeated *image*" sequences directly from a folder of images (instead of synthetic random vectors), optionally re-embedding a repeated image through a **fresh random augmentation** instead of showing the identical embedding. | `data.GenRecogClassifyData_AD`, `GenRecogClassifyData_AD_Batched`, `GenRecogClassifyData_AD_Precompute` (also duplicated/extended in `implementing_GenRecogClassifyData_AD.py`) |
| **Precomputed embedding banks** | One-time offline pass that embeds every image once (no-augmentation "base" bank) and `K` times with random augmentations ("aug" bank), at a chosen ResNet18 depth, so training doesn't re-run the CNN forward pass every step. | `data.build_precomputed_embedding_banks`, driver script `precompute_augmented_embeddings.py` |
| **Training scripts** | Train the featurizer alone, or jointly with the Hebbian core, with/without augmentation, at different ResNet depths. | `train_feature_layer_only.py`, `train_feature_layer_and_hebbff.py`, `train_feature_layer_and_hebbff_augment.py`, `train_feature_layer_and_hebbff_augment_extended_featurizer.py` |
| **Generalization (perf-vs-R) evaluation** | Sweeps test-time `R` and records accuracy / true-positive / false-positive rate, for a trained checkpoint evaluated on held-out image sets. | `plotting.get_recog_positive_rates`, `plotting.get_generalization`; driver `evaluate_hebbff_on_embeddings.py` |
| **FLD readout / `W_FLD`** | Fits the optimal linear (Fisher) readout of the hidden layer for familiar-vs-novel, given hidden activity `h` and the familiarity label — same tool the original paper used to fit a linear decoder to macaque IT population recordings (`plot_figs.py`, Fig. 7). Here it is re-purposed to read out the **model's own hidden layer**. | `plotting.compute_FLD_weights`, `plotting.get_pop_size_acc` |
| **Category decodability** | Trains a linear (logistic-regression) classifier to predict image category from (a) the raw 512-d ResNet embedding and (b) the network's Hebb-featurized intermediate embedding, on real photographs organized by category. | `test_hebb_category.py`, `test_augmentation_loader.py` |
| **Real image / category stimulus set** | Precomputed 512-d ResNet18 embeddings for a continuous-recognition-memory image set ("new/old/delay"), organized into 3 sub-experiments, each with several object-category subfolders. Used by both the perf-vs-R and category-decoding analyses. | `Embeddings/newolddelay{,2,3}/<category>/*.npy` (596 files total) |
| **Exploratory notebooks** | Scratch analysis / plotting used while developing the above. | `test_hebb_feature.ipynb`, `plot_augmentations.ipynb`, `scratchpad.ipynb` |

---

## 3. File-by-file guide

### Core library (imported by everything else)
- **`net_utils.py`** — `NetworkBase`/`StatefulBase` base classes: training loop
  (`_train_epoch`), curriculum training (`train_curriculum`, `train_multiR_curriculum`),
  checkpoint save/load (`.save()` auto-versions filenames as `name_(2).pkl`,
  `name_(3).pkl`, ... so re-running a training script never overwrites a prior run —
  `overwrite=False` is the default), `load_from_file()`, weight init helpers, accuracy
  functions.
- **`networks.py`** — All network classes:
  - `HebbNet` — the base Hebbian familiarity network described above.
  - `HebbFeatureLayer(HebbNet)` — adds a trainable `featurizer: Linear(Nx, d)` (or
    `Linear`+`ReLU`) in front of the Hebbian core, so the whole model is
    `image embedding (Nx-d) → featurizer → HebbFF(d, Nh, 1)`. Used with
    **precomputed** embeddings (fast — no CNN forward pass per training step).
  - `HebbExtendedFeatureLayer(HebbNet)` — same idea, but also carries an
    `embedding_model` (e.g. ResNet18 layers 4 + avgpool) so the *later* CNN layers can
    be fine-tuned jointly with the featurizer, taking a frozen earlier-layer
    (e.g. `layer3`) feature map as input. Used by the "extended featurizer" training
    script.
  - `HebbClassify(HebbNet)` — dual-head variant from the base paper (familiarity +
    category output in one net); kept for compatibility with `plot_figs.py` Fig. 7 but
    not used by the augmentation experiments in this repo.
  - `DetHebb` / `DetHebbNoSplit` — analytic/idealized model from the base paper (not
    used by the new results below).
- **`data.py`** — Data generators:
  - `generate_recog_data` / `generate_recog_data_batch` — original synthetic
    binary-vector recognition-memory task.
  - `GenRecogClassifyData` — samples repeat/novel sequences from a fixed
    `TensorDataset` of pre-embedded images (no augmentation).
  - `GenRecogClassifyData_AD` — image-folder version: loads images from disk, applies
    an `albumentations` transform (`RandomRotate90`, `HorizontalFlip`,
    `RandomResizedCrop`, `RandomBrightnessContrast` by default) with probability
    `1-P2`, embeds with a frozen ResNet18, per-timestep. `y=1` marks a repeat
    (whether or not it was re-augmented on presentation).
  - `GenRecogClassifyData_AD_Batched` — same, with an LRU image/embedding cache and
    batched CNN forward passes (much faster for repeated experiments over a fixed
    image pool).
  - `build_precomputed_embedding_banks` — one-time offline embedding of every image
    with no augmentation (`base_bank`, shape `[N,512]`) and `num_aug` augmented copies
    each (`aug_bank`, shape `[N,K,512]`), at a chosen truncation depth of ResNet18
    (`None`=avgpool/512-d, `"layer3"`=256-d, `"layer3_no_pool"`=256×14×14 flattened,
    `"layer2"`=128-d).
  - `GenRecogClassifyData_AD_Precompute` — reads the two banks above and generates
    training sequences by indexing into them (no CNN forward pass at train time) —
    this is what all the `_augment*` training scripts use.
  - `check_recognition_data`, `recog_chance`, `data_to_tensor` — sanity/helper utils.
- **`implementing_GenRecogClassifyData_AD.py`** — development copy of the
  `GenRecogClassifyData_AD*` classes plus a runnable demo at the bottom that loads one
  random category folder from a "newolddelay" stimulus directory and prints a sample
  batch; kept as a worked example of how the generator is wired up. (Hardcodes
  `ROOT_DIR = "/home/raaghav/hebbff/Images/Stimuli/newolddelay/"` — update this path
  for your machine before running it directly.)
- **`plotting.py`** — All evaluation/plotting utilities:
  - `get_recog_positive_rates(net, gen_data, ...)` — sweeps `R` (log- or
    linear-spaced) and returns `(R, accuracy, true_pos_rate, false_pos_rate)`. This is
    the core routine behind every "performance vs. R" plot.
  - `get_generalization(fname, ...)` — same, but memoizes the result to
    `<fname>_generalization.pkl` so repeated plotting doesn't re-run evaluation.
  - `plot_generalization(...)` — the standard 2-panel (accuracy; TP/FP rate) vs. `R`
    figure.
  - `compute_FLD_weights(h, isFam, neuronsMask=None, SCC=False, dPrime=False)` — given
    hidden-layer activity `h` `[T,N]` and a boolean familiarity label `isFam` `[T]`,
    returns the Fisher linear discriminant weight vector `w` (`W_FLD`, length `N`) and
    bias `b` that best separate familiar vs. novel hidden states (or, if `SCC=True`,
    the "simple/summed correlation classifier" baseline instead of the full FLD).
  - `get_pop_size_acc(fname, net, gen_data, R=None)` — ranks hidden units by `|W_FLD|`
    and measures recognition accuracy as a function of how many top units are kept, for
    both the `FLD` and `SCC` decoders (memoized to `<fname>_popSizeAcc.pkl`).
  - `get_evaluation_result`, `plot_hidden_activity`, `plot_weight`,
    `plot_familiar_output_distr`, `plot_RT_curves`, `plot_R_curriculum`,
    `plot_distribution`, `plot_corr_per_timepoint`, `get_corr_Wxb_Ax`,
    `fit_idealized`, `plot_and_fit` — figure-generation helpers used by
    `plot_figs.py` for the original paper's mechanism/idealized-model/capacity
    figures; not required for the 4 key results below but available if you want to
    reproduce the paper's original mechanistic analyses on your trained nets.
- **`bogacz.py`** — Reimplementation of the Bogacz & Brown (2003) plasticity model,
  used by the original paper only for the validation figure in `plot_figs.py`.
- **`dt_utils.py`** — Small generic utilities (`Timer`, `SaveAs` context manager,
  `subplots_square`, `numerical_gradient`); no project-specific logic.

### Training scripts (run these to produce checkpoints)
- **`train_feature_layer_only.py`** — Loads a **pretrained, frozen** synthetic-data
  `HebbNet` checkpoint (`publish/conv/HebbNet[50,16,1]_...pkl`, freezes `net.w1`), and
  trains **only the featurizer** (`Nx=512 → d=50`) on the Brady & Oliva (2008)
  "Unique Objects" image set (`publish/conv/BradyOliva2008_UniqueObjects_ResNet18.pkl`,
  precomputed non-augmented embeddings, no on-the-fly augmentation) via curriculum
  training. Saves to `results/HebbFeature_freezehebbff.pkl`.
- **`train_feature_layer_and_hebbff.py`** — Same setup, but the **whole** network
  (featurizer + Hebbian core) is trainable, still on the same non-augmented Brady &
  Oliva embeddings. Saves to `results/HebbFeature_no_freeze.pkl`. **This is the
  "non-augmented" reference model** used for the augmented-vs-non-augmented
  comparisons below (`Nx=512, d=50, Nh=16`).
- **`train_feature_layer_and_hebbff_augment.py`** — Jointly trains featurizer +
  Hebbian core using `GenRecogClassifyData_AD_Precompute` on **precomputed
  base+augmentation embedding banks** (`P2` controls how often a repeat is
  re-embedded through a random augmentation instead of shown identically; the script
  sets `P2=0`, i.e. *always* augment on repeat). Uses a larger network
  (`Nx=256, d=128, Nh=32`, ResNet `layer3`+avgpool features) and a real photograph set
  ("newolddelay" stimuli under `IMAGE_DIR`). Saves to
  `results/HebbFeature_no_freeze_augment_rutis.pkl`. **This is the "augmented" model.**
- **`train_feature_layer_and_hebbff_augment_extended_featurizer.py`** — Same
  augmented-training idea, but keeps ResNet's `layer4` + avgpool **trainable** as part
  of the featurizer (`HebbExtendedFeatureLayer`), taking the frozen `layer3`
  (`256×14×14`) feature map as the precomputed input instead of the pooled 512-d
  vector. Uses minibatches (`train_minibatch_size=16`) since this network is bigger.
  Saves to `results/HebbFeature_no_freeze_augment_extended_rutis.pkl`.
- **`script_simple.py`** — Minimal, from-scratch example of training a plain
  `HebbNet`/`nnLSTM` on synthetic binary-vector data; useful as a sanity check that
  your environment/training loop works before running the image-based scripts above.

All training scripts print `net.hist['increment_R']` progress to stdout as the
curriculum advances `R`, and write TensorBoard logs to their `logs`/`folder` path
(view with `tensorboard --logdir results/tensor_board_logs_no_freeze_augment_rutis`).

### Evaluation / analysis scripts (run these on trained checkpoints)
- **`evaluate_hebbff_on_embeddings.py`** — Loads a `HebbFeatureLayer` checkpoint and,
  for every experiment subfolder under `Embeddings/`, builds a `GenRecogClassifyData`
  sample space from that folder's `.npy` embeddings and computes the accuracy /
  true-positive-rate / false-positive-rate curve vs. `R` (`get_recog_positive_rates`).
  Saves a `.pkl` + `.json` with the raw curves and a `.png` overlay plot comparing all
  experiments. This is the **"performance vs. R"** driver — run it once per
  checkpoint (augmented and non-augmented) to get the two curves to compare.
- **`test_hebb_category.py`** — For every experiment folder under `Embeddings/`
  (each containing category subfolders of `.npy` embeddings): stratified
  train/test split by category, standardizes features, then trains a **linear
  softmax classifier** to predict category from (a) the raw 512-d embedding and
  (b) the embedding passed through the trained network's `featurizer` (loaded
  directly from the checkpoint's `state_dict`, i.e. the network's **intermediate**
  representation before the Hebbian core). Reports train/test accuracy for both,
  their difference, and saves a grouped bar chart
  (`results/hebb_category_comparison_barplot.png`). This is the **"category
  prediction from intermediate models"** driver.
- **`test_augmentation_loader.py`** — Thin sanity-check script (imports `data`, no
  executable logic) used while debugging the augmentation data loader; not part of the
  reproduction pipeline.
- **`precompute_augmented_embeddings.py`** — Driver for
  `data.build_precomputed_embedding_banks`: walks an image directory, embeds every
  image once (base bank) and `num_augmentations_per_image` times with random
  augmentations (aug bank) at a chosen ResNet depth, and writes both to
  `precomputed_embeddings/{base_bank,aug_bank}/*.npy`. **Run this before any
  `_augment*` training script** that reads from those paths.

### Data / pretrained assets
- **`Embeddings/`** — Precomputed 512-d ResNet18 embeddings (`.npy`, one file per
  image) for a continuous-recognition-memory "new/old/delay" image set, split into
  three sub-experiments, each with category subfolders:
  - `newolddelay/`: `houses`(44), `landscapes`(43), `mobility`(40), `phones`(40),
    `smallAnimal`(42), `ztrain`(6, held-out/practice items).
  - `newolddelay2/`: `fruit`(31), `kids`(30), `military`(35), `space`(30),
    `zzanimal`(30).
  - `newolddelay3/`: `1cars`(44), `2food`(44), `3people`(45), `4spatial`(45),
    `5animals`(47).

  These are consumed directly by `evaluate_hebbff_on_embeddings.py` (performance vs.
  R, per experiment) and `test_hebb_category.py`/`test_hebb_feature.ipynb`
  (category decodability, using the subfolder name as the class label).
- **`publish/`** — Assets and pretrained checkpoints carried over from the original
  HebbFF paper's public release (`publish/RNN`, `publish/antiHebb`, `publish/bogacz`,
  `publish/capacity`, `publish/mechanism`, `publish/idealized`, `publish/inferotemporal`,
  `publish/conv`). The pieces this project actually depends on are inside
  `publish/conv/`:
  - `BradyOliva2008_UniqueObjects_ResNet18.pkl` — precomputed non-augmented 512-d
    ResNet18 embeddings for the Brady & Oliva (2008) "Unique Objects" memorability
    image set (used by `train_feature_layer_only.py` /
    `train_feature_layer_and_hebbff.py`).
  - `HebbNet[50,16,1]_train=cur1_incr=plus1_w1init=randn_b1init=scalar_w2init=scalar.pkl`
    — a synthetic-data-pretrained `HebbNet` used as the frozen Hebbian core in
    `train_feature_layer_only.py`.
- **Not included in the repo (gitignored, must be provided/generated locally)**:
  `Images/` (raw image files for `_AD`/`_augment` training — point `IMAGE_DIR` at your
  own copy of the "newolddelay" or other stimulus set), `precomputed_embeddings/`
  (output of `precompute_augmented_embeddings.py`), `results/` (all trained
  checkpoints, logs, and evaluation outputs land here).

### Notebooks
- **`test_hebb_feature.ipynb`** — Interactive scratchpad for loading a
  `HebbFeatureLayer` checkpoint, plotting its loss/accuracy curve, curriculum
  trajectory, generalization curve, output distributions, reaction-time curves, and
  weight heatmaps; also contains a cell that loads and concatenates all `Embeddings/`
  experiment folders into one `(x, y, class_names)` tensor set (same loader as
  `test_hebb_category.py`) for ad hoc exploration.
- **`plot_augmentations.ipynb`** — Visualizes example `albumentations` transforms
  applied to sample stimulus images (sanity-check that the augmentation pipeline
  looks reasonable).
- **`scratchpad.ipynb`** — Misc one-off checks (e.g. listing/loading image files).

### `archive/`
Frozen copies of the original HebbFF paper's scripts (`script_*.py`, `plot_*.py`,
older `data.py`/`networks.py`/`plotting.py`, an old CPU/numpy fast-weights
implementation, etc.). Not maintained and not needed to reproduce anything below —
kept only as historical reference / for cross-checking behavior against the upstream
implementation.

---

## 4. Setup

```bash
# Python 3.10 recommended (matches development environment)
pip install torch torchvision numpy scipy matplotlib seaborn joblib albumentations pillow
```

A CUDA GPU is strongly recommended for `precompute_augmented_embeddings.py` (embeds
every image through ResNet18 up to `num_augmentations_per_image` times) and for
`train_feature_layer_and_hebbff_augment_extended_featurizer.py` (fine-tunes ResNet18's
`layer4`); the other training/evaluation scripts run fine on CPU, just slower.

**Before running anything**, note that several scripts hardcode absolute paths from
the original author's machine and must be edited first:
- `IMAGE_DIR` / `ROOT_DIR` in `precompute_augmented_embeddings.py`,
  `train_feature_layer_and_hebbff_augment*.py`, and
  `implementing_GenRecogClassifyData_AD.py` (currently
  `/home/raaghav/hebbff/Images/...`) — point these at your local copy of the image
  stimulus set.
- `base_bank` / `aug_bank` paths must match between
  `precompute_augmented_embeddings.py` (writer) and the training scripts (reader) —
  they must use the same ResNet `layer` argument (`None`, `"layer3"`,
  `"layer3_no_pool"`, or `"layer2"`) so embedding dimensions line up.
- `--checkpoint` defaults in `evaluate_hebbff_on_embeddings.py` and
  `test_hebb_category.py` point at specific `results/...pkl` filenames from a prior
  run — pass `--checkpoint <your file>` explicitly (checkpoint filenames are
  auto-suffixed `_(2)`, `_(3)`, ... by `net.save()`, so the exact name depends on how
  many times you ran the training script).

---

## 5. Reproducing the key results

All four results below compare (at least) two trained checkpoints:
- **Non-augmented model**: output of `train_feature_layer_and_hebbff.py`
  (`Nx=512, d=50, Nh=16`, trained on static Brady & Oliva embeddings, no augmentation).
- **Augmented model**: output of `train_feature_layer_and_hebbff_augment.py`
  (`Nx=256, d=128, Nh=32`, trained with random-augmentation-on-repeat via the
  precomputed aug bank).

You can substitute the extended-featurizer checkpoint
(`train_feature_layer_and_hebbff_augment_extended_featurizer.py`) as a second
"augmented" variant if you want a 3-way comparison; adjust `Nx`/`d`/`Nh` in the
snippets below to match whichever checkpoint you load (see the table in each
script's header for its exact dimensions).

### Step 0 — Precompute embeddings and train both models

```bash
# 1) Precompute base + augmented embedding banks for the augmented-training pipeline
python precompute_augmented_embeddings.py        # writes precomputed_embeddings/{base_bank,aug_bank}/*.npy

# 2) Train the non-augmented reference model
python train_feature_layer_and_hebbff.py          # -> results/HebbFeature_no_freeze.pkl

# 3) Train the augmented model
python train_feature_layer_and_hebbff_augment.py  # -> results/HebbFeature_no_freeze_augment_rutis.pkl
```

Each writes `net.hist['increment_R']` (the curriculum's `R` trajectory) and
`train_loss`/`train_acc` into the checkpoint's `hist` dict, and TensorBoard event
files under `logs`.

### Result 1 — Performance (accuracy / TP / FP) as a function of R, augmented vs. non-augmented

```bash
python evaluate_hebbff_on_embeddings.py \
    --checkpoint results/HebbFeature_no_freeze.pkl \
    --out results/embedding_eval_no_augment.json

python evaluate_hebbff_on_embeddings.py \
    --checkpoint results/HebbFeature_no_freeze_augment_rutis.pkl \
    --out results/embedding_eval_augment.json
```

Each run evaluates the checkpoint on every `Embeddings/<experiment>` folder
(sweeping `R` logarithmically via `plotting.get_recog_positive_rates`) and saves a
per-experiment accuracy/TP-rate/FP-rate-vs-`R` curve plus an overlay plot
(`*_hebbfeature_*.png`). 

### Result 2 — Mean `W_FLD` for augmented vs. non-augmented models

The Fisher discriminant readout weight over the hidden layer (`W_FLD`) is computed by
`plotting.compute_FLD_weights(h, isFam)`, where `h` is hidden-layer activity and
`isFam` is the ground-truth familiarity label — this is the same routine the original
paper used to fit a linear familiarity decoder onto IT population recordings
(`plot_figs.py`, Fig. 7b: `w2, b2 = plotting.compute_FLD_weights(db['h'], isFam)`).
Here it's applied to the trained model's own hidden units instead of neural data:

```python
import numpy as np, torch
from net_utils import load_from_file
from data import GenRecogClassifyData
import plotting

def mean_wfld_for_checkpoint(ckpt_path, embeddings_pkl, Rs=None):
    net = load_from_file(ckpt_path)
    images = torch.load(embeddings_pkl)              # [N, Nx] pooled embeddings
    sample_space = torch.utils.data.TensorDataset(images, torch.zeros(images.shape[0], 1))
    generator = GenRecogClassifyData(sampleSpace=sample_space)

    if Rs is None:
        Rs = np.unique(np.logspace(0, 2, 20, dtype=int))   # R = 1 ... 100
    data = generator(T=len(images), R=list(Rs), P=0.5, batchSize=None, multiRep=False)

    db = net.evaluate_debug(data.tensors, recogOnly=True)
    isFam = (db['data'].tensors[1] == 1).bool().squeeze()
    w_fld, b_fld = plotting.compute_FLD_weights(db['h'], isFam)
    return w_fld  # length-Nh vector of per-unit FLD readout weights

w_noaug = mean_wfld_for_checkpoint(
    "results/HebbFeature_no_freeze.pkl",
    "publish/conv/BradyOliva2008_UniqueObjects_ResNet18.pkl")
w_aug = mean_wfld_for_checkpoint(
    "results/HebbFeature_no_freeze_augment_rutis.pkl",
    "<precomputed non-augmented base-bank embeddings for the augmented model's stimulus set>")

print("Non-augmented: mean |W_FLD| = {:.4f} (n={} units)".format(np.abs(w_noaug).mean(), len(w_noaug)))
print("Augmented:     mean |W_FLD| = {:.4f} (n={} units)".format(np.abs(w_aug).mean(), len(w_aug)))
```


### Result 3 — Category prediction from intermediate representations

```bash
python test_hebb_category.py \
    --checkpoint results/HebbFeature_no_freeze_augment_rutis.pkl \
    --embeddings-root Embeddings \
    --out results/hebb_category_comparison_augmented.json

python test_hebb_category.py \
    --checkpoint results/HebbFeature_no_freeze.pkl \
    --embeddings-root Embeddings \
    --out results/hebb_category_comparison_no_augment.json
```

For each `Embeddings/<experiment>` folder, this loads the checkpoint's `featurizer`
weights directly out of its `state_dict` (`featurizer.weight`/`featurizer.bias` —
i.e., **without** needing to reconstruct the full `HebbFeatureLayer`, so it works even
if `Nx`/`d` differ between checkpoints), does a stratified 80/20 train/test split by
category, standardizes features, and trains a linear softmax classifier on:
- **`raw_512d`** — the untouched ResNet18 embedding, and
- **`hebb_featurized_<d>d`** — the same embedding passed through the trained
  featurizer (the network's learned, bottlenecked "intermediate" representation, i.e.
  what actually feeds the Hebbian familiarity core).

It prints and saves `test_acc` for both, plus `test_acc_delta = featurized − raw`,
per experiment, and a grouped bar chart
(`results/hebb_category_comparison_<...>_barplot.png`). Run it once per checkpoint
(as above) and compare the `hebb_featurized` bars between the two output JSONs to see
whether augmentation changes how much category information survives the
familiarity-network bottleneck relative to the shared `raw_512d` baseline.

### Result 4 — Memorability vs. category decodability, across models

This combines the per-image outputs of Results 1–3: define an image's
**memorability** from the network's recognition behavior, and its **category
decodability** from a linear classifier's per-item margin/confidence, then correlate
the two — repeated for each model (non-augmented, augmented, and/or raw-ResNet
baseline) to see whether the relationship holds or changes.

```python
import numpy as np, torch
from pathlib import Path
from net_utils import load_from_file
from data import GenRecogClassifyData
from test_hebb_category import load_experiment, load_featurizer_from_checkpoint, apply_featurizer

def per_item_scores(ckpt_path, experiment_dir, R=10, n_repeats=50):
    net = load_from_file(ckpt_path)
    x, y, class_names, _ = load_experiment(experiment_dir)     # x: [N,512] embeddings, y: category id

    # --- Memorability: fraction of trials each image is correctly reported "familiar"
    #     when it repeats at lag R (averaged over many random sequence draws).
    sample_space = torch.utils.data.TensorDataset(x, torch.zeros(len(x), 1))
    generator = GenRecogClassifyData(sampleSpace=sample_space)
    hits = np.zeros(len(x)); shown = np.zeros(len(x))
    for _ in range(n_repeats):
        data = generator(T=len(x), R=R, P=0.5, batchSize=None, multiRep=False)
        idxData, yData = data.tensors
        out = net.evaluate(data.tensors) if hasattr(net, "evaluate") else net(idxData)
        isFam = (yData[:, 0] == 1)
        # match repeated items back to their source row in x by nearest-neighbor on the embedding
        for t in torch.where(isFam)[0].tolist():
            src = ((x - idxData[t]).abs().sum(dim=1)).argmin().item()
            shown[src] += 1
            hits[src] += float(out[t] > 0.5)
    memorability = np.divide(hits, shown, out=np.full_like(hits, np.nan), where=shown > 0)

    # --- Category decodability: per-item margin from a linear probe on the featurized embedding
    featurizer = load_featurizer_from_checkpoint(ckpt_path)
    feat_x = apply_featurizer(featurizer, x)
    clf = torch.nn.Linear(feat_x.shape[1], len(class_names))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        logits = clf(feat_x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = clf(feat_x)
        true_logit = logits.gather(1, y[:, None]).squeeze(1)
        other_logit = logits.clone().scatter_(1, y[:, None], -float("inf")).max(dim=1).values
        decodability = (true_logit - other_logit).numpy()   # per-item classification margin

    return memorability, decodability, class_names

mem, dec, _ = per_item_scores("results/HebbFeature_no_freeze_augment_rutis.pkl", "Embeddings/newolddelay")
mask = ~np.isnan(mem)
print("corr(memorability, category decodability) =", np.corrcoef(mem[mask], dec[mask])[0, 1])
```

Run this for each checkpoint you want to compare (non-augmented, augmented,
extended-featurizer) and, as a raw-embedding control, repeat the decodability half
with `feat_x = x` (skip the featurizer) to see whether any observed
memorability–decodability relationship is a property of the raw image statistics or
is introduced/altered by the trained network's bottleneck. Plot `memorability` vs.
`decodability` as a scatter per model (colored by category) and report the Pearson/
Spearman correlation for each — a positive correlation would indicate that images
whose category is easier to read out of the network's intermediate layer are also
the images the network is more likely to correctly recognize as familiar.

---