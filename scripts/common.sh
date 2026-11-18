# Shared settings of the launch scripts (sourced). Every script runs from the repository root, on one GPU
# (GPU=<id>, default 0), with the Python interpreter $PY (default: .venv/bin/python if it exists, else python),
# and writes its logs to logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${GPU:-0} USE_TF=0
if [ -z "$PY" ]; then [ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python; fi
mkdir -p logs

DATASETS="celeba fairface acs sent"     # the four label-skew settings
SEEDS="42 43 44"                        # partition seeds

# Partition arguments of a label-skew setting: part <dataset> <seed>
part() {
  case $1 in
    celeba)   echo "images --dataset celeba --max-samples 0 --num-clients 100 --partition-on label --partition-seed $2" ;;
    fairface) echo "images --dataset fairface --max-samples 80000 --num-clients 50 --partition-on label --partition-seed $2" ;;
    acs)      echo "tabular --task ACSIncome --partition label --partition-seed $2" ;;
    sent)     echo "tabular --task Sent140 --partition label --tab-root conv_results_tabular_sent140_label --partition-seed $2" ;;
  esac
}
# All encoders of a dataset, and the main one (tau/lambda grid, ablations, inference audit)
encs() { case $1 in celeba|fairface) echo "sd-vae dinov2-small clip-b32 resnet18" ;; acs) echo raw ;; sent) echo "minilm mpnet" ;; esac; }
main_enc() { case $1 in celeba|fairface) echo clip-b32 ;; acs) echo raw ;; sent) echo minilm ;; esac; }
