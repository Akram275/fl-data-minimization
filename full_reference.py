"""
Full-federation reference: FedAvg on ALL clients and centralized training on all clients' records (the union of
their FL training splits), with the partition, model, FL budget and test pool of run_protocol.py. It does not
depend on the encoder or the selection, so there is one run per partition.

  python full_reference.py images --dataset celeba --max-samples 0 --num-clients 100 --partition-on label --partition-seed 42
  python full_reference.py tabular --task ACSIncome --partition states
Outputs: <out>/<partition root>/full/{fl,cen}.csv
"""
import argparse
from pathlib import Path

import src.federated_learning.torch_fl as fl
from src.experiment import add_partition_args, load_partition


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_partition_args(p, encoders=[])
    p.add_argument('--rounds', type=int, default=20)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--out', default="results")
    args = p.parse_args()
    part = load_partition(args)
    out = Path(args.out) / part.root / "full"
    out.mkdir(parents=True, exist_ok=True)
    idx = list(range(len(part.datasets)))
    if not (out / "fl.csv").exists():
        fl.save_scores_to_csv(fl.run_exp(idx, part.datasets, part.test_x, part.test_y, epochs=args.local_epochs,
                                         max_iterations=args.rounds, aggregation_method='fedavg', lr_decay=True),
                              out / "fl.csv")
    if not (out / "cen.csv").exists():
        fl.save_scores_to_csv(fl.run_pooled_training(part.datasets, part.test_x, part.test_y, args.rounds),
                              out / "cen.csv")
    print(f">> done {out}")


if __name__ == '__main__':
    main()
