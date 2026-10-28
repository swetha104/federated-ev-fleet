"""Non-federated baselines on exactly the same per-client splits as the FL runs.

- local: each client trains only on its own data (v1's "personalized" baseline).
- centralized: all clients' training data pooled in one place; the upper bound
  FL is trying to approach without moving data (v1 had no such baseline).

    python -m evfl.baselines --kind ev --num-clients 50 --epochs 10 --out-dir outputs/ev-baselines
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from evfl.data import InsufficientDataError, client_ids, client_loaders, target_scale
from evfl.model import build_model
from evfl.train import evaluate, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["ev", "cs"], default="ev")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs/baselines")
    p.add_argument("--num-clients", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10, help="match rounds x local epochs of the FL run")
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    scale = target_scale(args.kind)
    loaders = {}
    for cid in client_ids(args.kind, data_dir)[: args.num_clients]:
        try:
            loaders[cid] = client_loaders(args.kind, cid, data_dir, args.batch_size)
        except InsufficientDataError:
            pass  # same clients the FL run cannot use

    torch.manual_seed(args.seed)
    local = {}
    for cid, (train_loader, test_loader) in loaders.items():
        model = build_model(args.kind)
        train(model, train_loader, args.epochs, args.lr)
        local[cid] = evaluate(model, test_loader, scale)

    torch.manual_seed(args.seed)
    pooled = DataLoader(ConcatDataset([tr.dataset for tr, _ in loaders.values()]), batch_size=args.batch_size, shuffle=True)
    central_model = build_model(args.kind)
    train(central_model, pooled, args.epochs, args.lr)
    central = {cid: evaluate(central_model, te, scale) for cid, (_, te) in loaders.items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps({"config": vars(args), "local": local, "centralized": central}, indent=2))
    print(f"Results: {out}/baselines.json ({len(loaders)} usable clients)")


if __name__ == "__main__":
    main()
