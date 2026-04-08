import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lc2.model import LC2Model
from lc2.losses import BidirectionalInfoNCELoss
from lc2.data.train_dataset import build_vivid_phase1_pool, ContrastivePairMiner
from lc2.data.transforms import (
    get_transform,
    range_to_normalized_disparity,
    depth_to_normalized_disparity,
    crop_range_to_camera_fov,
    squeeze_depth,
)


def load_input(pool, idx, transform, hfov, forward_col_frac):
    data = np.load(str(pool.paths[idx]))
    if data.ndim > 2:
        data = squeeze_depth(data)
    if pool.is_range[idx]:
        data = crop_range_to_camera_fov(data, camera_hfov_deg=hfov, forward_col_frac=forward_col_frac)
        data = range_to_normalized_disparity(data)
    else:
        data = depth_to_normalized_disparity(data)
    return transform(data)


class OneToOneDataset(Dataset):
    def __init__(self, pairs, pool, transform, hfov, forward_col_frac):
        self.pairs = pairs
        self.pool = pool
        self.transform = transform
        self.hfov = hfov
        self.forward_col_frac = forward_col_frac

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        r, d, psi = self.pairs[idx]
        return {
            "image_i": load_input(self.pool, r, self.transform, self.hfov, self.forward_col_frac),
            "image_j": load_input(self.pool, d, self.transform, self.hfov, self.forward_col_frac),
            "is_range_i": True,
            "is_range_j": False,
            "psi": float(psi),
            "range_idx": int(r),
            "depth_idx": int(d),
        }


@torch.no_grad()
def eval_one_to_one(model, dataset, device):
    model.eval()
    all_ranges = []
    all_depths = []
    range_ids = []
    depth_ids = []
    for sample in dataset:
        img_r = sample["image_i"].unsqueeze(0).to(device)
        img_d = sample["image_j"].unsqueeze(0).to(device)
        desc_r = model(img_r, torch.tensor([True], device=device)).cpu().squeeze(0)
        desc_d = model(img_d, torch.tensor([False], device=device)).cpu().squeeze(0)
        all_ranges.append(desc_r)
        all_depths.append(desc_d)
        range_ids.append(sample["range_idx"])
        depth_ids.append(sample["depth_idx"])
    R = torch.stack(all_ranges)
    D = torch.stack(all_depths)
    R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-8)
    D = D / D.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims = R @ D.T
    top1 = sims.argmax(dim=1)
    correct = (top1 == torch.arange(len(dataset))).sum().item()
    return {
        "correct": int(correct),
        "total": int(len(dataset)),
        "r1": float(correct / len(dataset)),
        "diag_mean": float(torch.diag(sims).mean().item()),
        "offdiag_mean": float(((sims.sum() - torch.diag(sims).sum()) / (sims.numel() - len(dataset))).item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sequence", default="campus_day1")
    ap.add_argument("--depth_cache_dir", required=True)
    ap.add_argument("--range_cache_dir", required=True)
    ap.add_argument("--range_subsample", type=int, default=30)
    ap.add_argument("--depth_subsample", type=int, default=30)
    ap.add_argument("--spatial_radius", type=float, default=100.0)
    ap.add_argument("--camera_hfov_deg", type=float, default=85.9)
    ap.add_argument("--forward_col_frac", type=float, default=0.996)
    ap.add_argument("--resize_h", type=int, default=240)
    ap.add_argument("--resize_w", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--freeze_until", type=int, default=0)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transform((args.resize_h, args.resize_w))

    pool = build_vivid_phase1_pool(
        root=args.root,
        sequences=[args.sequence],
        depth_cache_dir=args.depth_cache_dir,
        range_cache_dir=args.range_cache_dir,
        range_subsample=args.range_subsample,
        depth_subsample=args.depth_subsample,
        n_crops=1,
        crop_fov_deg=args.camera_hfov_deg,
        camera_hfov_deg=args.camera_hfov_deg,
        spatial_radius=args.spatial_radius,
    )
    miner = ContrastivePairMiner(pool=pool, max_range_m=50.0, camera_fov_deg=args.camera_hfov_deg, positive_mode="cross_modal_only")
    pos_arr = pool.position_array()
    grouped = {}
    for i, j, psi in miner.pairs:
        if psi <= 0:
            continue
        if pool.is_range[i] and not pool.is_range[j]:
            r, d = i, j
        elif pool.is_range[j] and not pool.is_range[i]:
            r, d = j, i
        else:
            continue
        dist = float(np.linalg.norm(pos_arr[r] - pos_arr[d]))
        grouped.setdefault(r, []).append((d, dist, float(psi)))
    pairs = []
    for r in sorted(grouped):
        d, dist, psi = sorted(grouped[r], key=lambda x: x[1])[0]
        pairs.append((int(r), int(d), float(psi)))

    dataset = OneToOneDataset(pairs, pool, transform, args.camera_hfov_deg, args.forward_col_frac)
    bs = min(len(dataset), 32)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

    model = LC2Model(num_clusters=16, encoder_dim=512, vladv2=False, pooling="gem", freeze_until=args.freeze_until).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
    criterion = BidirectionalInfoNCELoss(temperature=args.temperature)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

    history = []
    best = {"epoch": -1, "r1": -1.0}
    init_eval = eval_one_to_one(model, dataset, device)
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            img_i = batch["image_i"].to(device)
            img_j = batch["image_j"].to(device)
            ir_i = batch["is_range_i"].to(device)
            ir_j = batch["is_range_j"].to(device)
            desc_i = model(img_i, ir_i)
            desc_j = model(img_j, ir_j)
            loss = criterion(desc_i, desc_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ev = eval_one_to_one(model, dataset, device)
        ev["epoch"] = epoch
        ev["loss"] = float(loss.item())
        history.append(ev)
        if ev["r1"] > best["r1"]:
            best = dict(ev)
            ckpt_dir = Path(args.output_json).parent / Path(args.output_json).stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "best_score": ev["r1"]}, ckpt_dir / "best.pth.tar")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:03d} loss={ev['loss']:.4f} R1={ev['correct']}/{ev['total']} ({ev['r1']*100:.1f}%) diag={ev['diag_mean']:.4f} offdiag={ev['offdiag_mean']:.4f}")

    out = {
        "pairs": [{"range_idx": r, "depth_idx": d, "psi": p} for r, d, p in pairs],
        "init_eval": init_eval,
        "best": best,
        "history": history,
        "config": vars(args),
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2))
    print(json.dumps({"init": init_eval, "best": best}, indent=2))


if __name__ == "__main__":
    main()
