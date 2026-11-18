# Client coresets: differentially private client selection for data minimization in FL

Code of the paper *Do We Need Everyone? Differentially Private Client Selection for Data Minimization in Federated
Learning*.

Before training, every client encodes a random sample of its records with a frozen public encoder and sends one
noisy kernel-mean sketch. The server groups statistically redundant clients with size-weighted k-medoids on the
resulting MMDs, keeps one representative per group (a *client coreset*), and trains only those with FedAvg,
weighted by the record mass of their groups. A drift penalty favors representatives close to the population, a
stopping rule chooses the number of clients, and the whole selection phase is record-level differentially
private.

## Layout

| Path | Content |
|---|---|
| `src/selection/protocol.py` | the selection protocol: sampled encoding, Round 0 shared space, Round 1 sketches, optional DP |
| `src/selection/embedding_selectors.py` | `ClientCoresetSelector`: weighted k-medoids, drift penalty, stopping rules and their DP calibration |
| `src/selection/dp_accounting.py` | noise calibration and RDP accounting of the selection phase (with Poisson-sampling amplification) |
| `src/selection/private_selection.py` | budget allocation, and the all-records variant of the protocol (cost benchmark) |
| `src/selection/ddem.py`, `fedbary.py` | the published pre-training selectors DDEM and FedBary |
| `src/selection/random_baselines.py` | random federations matched on client count and record volume |
| `src/embedding/` | frozen encoders (`encoders.py`) and the sketch primitives (`snapshots.py`) |
| `src/federated_learning/torch_fl.py` | FedAvg, FedProx, SCAFFOLD, DivFL and centralized training |
| `src/data/` | dataset loaders, ACS and Sent140 caches, Dirichlet partitions |
| `src/experiment.py` | loading one client partition for the experiment scripts |
| `run_protocol.py` | main pipeline: selection with every method, then FedAvg, centralized control and matched random federations |
| `full_reference.py` | FedAvg on all clients and centralized training on all records |
| `stopping_check.py` | coverage of the stopping rule (selection only) |
| `selection_checks.py` | robustness: quadratic penalty, random-feature and sample seeds (selection only) |
| `central_check.py` | why the coreset's records train below all records (centralized controls) |
| `mia_check.py` | distribution and membership inference on the coreset, optionally with DP-SGD training |
| `privacy_check.py` | independent check of the privacy accounting with Google's `dp_accounting` |
| `overhead_benchmark.py` | client and server cost of the protocol |
| `scripts/` | launchers of every experiment of the paper |
| `paper/` | generators of the paper's tables, numbers and figures |

## Installation

Python 3.10 and a CUDA GPU (the experiments used an NVIDIA L40S).

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt     # install the CUDA build of torch/torchvision for your driver
```

The launchers use `.venv/bin/python` when it exists (or `PY=...`), and the GPU given by `GPU=<id>` (default 0).
All commands run from the repository root.

## Data

Everything lives under `data/` (not versioned):

| Dataset | Expected files | Preparation |
|---|---|---|
| CelebA | `data/celeba-dataset/{list_attr_celeba.csv, list_eval_partition.csv, img_align_celeba/}` | Kaggle release of CelebA |
| FairFace | `data/FairFace/{train_labels.csv, val_labels.csv, train/, val/}` | FairFace (padding 0.25) |
| ACSIncome | folktables' 2018 1-Year person files in `data/2018/1-Year/` | `python -m src.data.acs_tabular --task ACSIncome` |
| Sent140 | `data/sentiment140/training.1600000.processed.noemoticon.csv` | `python -m src.data.sent140 build`, then `embed --encoder minilm` and `embed --encoder mpnet` |

Image encodings (`data/embeddings/`) and the 32x32 training arrays (`data/fl_arrays/`) are computed and cached on
first use. The encoders are downloaded from the Hugging Face Hub.

## Reproducing the paper

```bash
GPU=0 scripts/run_tuning.sh           # FedProx mu and SCAFFOLD lr on all clients (values fixed in run_main.sh)
GPU=0 scripts/run_main.sh             # selections and FL results, private selection, ablations  -> results/
GPU=0 scripts/run_full_reference.sh   # FedAvg on all clients, centralized on all records          -> results/
GPU=0 scripts/run_checks.sh           # coverage, robustness, centralized controls, accounting, cost -> checks/
GPU=0 scripts/run_mia.sh              # distribution and membership inference                     -> mia_check/
GPU=0 scripts/run_mia_dp.sh           # the same with DP-SGD training (eps2 = 8 and 1)             -> mia_check/
```

Then the tables, numbers and figures:

```bash
.venv/bin/python paper/make_tables.py > tables.txt   # LaTeX tables and number macros; fig_private, pareto_label
.venv/bin/python paper/make_mia_tables.py            # inference tables and in-text numbers
.venv/bin/python paper/make_figures.py               # sketch map of CelebA clients
```

| Paper | Source |
|---|---|
| label skew, attribute skew, census and text, natural clients, private selection, drift correctors, tau/lambda grid, stopping rules, sampling rate and sketch size | `run_main.sh`, `run_full_reference.sh` -> `make_tables.py` |
| coverage of the stopping rule | `run_checks.sh` (`stopping_check.py`) -> `make_tables.py` |
| why the coreset's records train below all records, step-matched column | `run_checks.sh` (`central_check.py`) -> `make_tables.py` |
| robustness of the configuration | `run_checks.sh` (`selection_checks.py`, output in `checks/robustness/`) |
| independent accounting check | `run_checks.sh` (`privacy_check.py`, output in `checks/privacy_check.txt`) |
| cost of selection | `run_checks.sh` (`overhead_benchmark.py`, output in `checks/overhead/`) |
| a representative hides in its group, and its appendix | `run_mia.sh`, `run_mia_dp.sh` -> `make_mia_tables.py` |

Runs are deterministic given the partition, sample, noise and model seeds (up to GPU nondeterminism in training,
a few tenths of a point). Every script skips the outputs that already exist, so an interrupted launcher can be
restarted.

## Notes

- **Round 0 sampling.** By default (`--gamma0` unset), Round 0 uses the Round 1 sample (rate gamma = 0.5) and the
  selection's DP noise is calibrated for that single sample; this is the configuration that produced the
  paper's results. `--gamma0 0.2` runs the two-sample design described in the paper, with an independent
  Round 0 sample of rate 0.2 and the matching calibration (`calibrate_two_samples`), which is what
  `privacy_check.py` accounts.
- **Random federations.** The matched random federations are seeded from the results path of the paper's runs
  (`RANDOM_SEED_ROOT` in `run_protocol.py`), so they do not depend on `--out`.
