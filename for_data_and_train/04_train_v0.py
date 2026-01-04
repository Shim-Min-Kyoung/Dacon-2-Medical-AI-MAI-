import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


'''
============================================================
[설정] 경로 및 하이퍼파라미터 (v0)
    - PCC 개선 + seed2025 + 100ep + epoch soup
    - full embedding 사용
    - train/val split만 사용 (test split 미사용)
    - 학습/전처리 데이터는 for_data_and_train/local_data/ 이하 사용
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"

EMB_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_evo2L26_emb_full.npz"
OUTPUT_DIR = PROJECT_ROOT / "ckpt_v0"
SEED = 2025

print(f"[System] Input Data: {EMB_PATH}")
print(f"[System] Output Dir: {OUTPUT_DIR}")
print(f"[System] Seed: {SEED}")

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {EMB_PATH}")


'''
============================================================
유틸리티
============================================================
'''
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=-1)


'''
============================================================
Dataset
    - .npz 내부의 ref_emb/var_emb/label/group/var_len/split 사용
    - split_idx로 train(0) / val(1) 선택
    - PCC 계산 안정화를 위해 group을 버킷팅(group_bucket_size)
============================================================
'''
class SharedEmbeddingDataset(Dataset):
    def __init__(self, data_dict: np.lib.npyio.NpzFile, split_idx: int, group_bucket_size: int = 100):
        super().__init__()
        split_arr = data_dict["split"].astype(np.int8)
        mask = (split_arr == split_idx)

        self.ref_emb = data_dict["ref_emb"][mask]
        self.var_emb = data_dict["var_emb"][mask]
        self.label   = data_dict["label"][mask].astype(np.int8)
        self.var_len = data_dict["var_len"][mask].astype(np.int16)
        raw_group    = data_dict["group"][mask].astype(np.int64)

        # PCC 그룹용 버킷팅
        self.group = raw_group // group_bucket_size

        # 입력 임베딩 차원
        self.H = self.ref_emb.shape[1]

    def __len__(self) -> int:
        return self.ref_emb.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.ref_emb[idx],
            self.var_emb[idx],
            int(self.label[idx]),
            int(self.group[idx]),
            int(self.var_len[idx]),
        )


'''
============================================================
DataLoader collate
    - numpy batch 형태로 쌓아서 반환 (학습 루프에서 torch.tensor로 변환)
============================================================
'''
def collate_emb(batch):
    ref_e, var_e, labels, groups, var_lens = zip(*batch)
    return (
        np.stack(ref_e),
        np.stack(var_e),
        np.array(labels, dtype=np.int64),
        np.array(groups, dtype=np.int64),
        np.array(var_lens, dtype=np.int64),
    )


'''
============================================================
Model Architecture (ResBlock 6)
    - ProjectionHead: 4096 -> 1024 -> 2048 (L2 normalize)
    - SiameseHead: ref/var 각각 동일 head 공유
============================================================
'''
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 4096, mid_dim: int = 1024, out_dim: int = 2048, n_blocks: int = 6):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        blocks = [ResidualBlock(mid_dim, dropout=0.1) for _ in range(n_blocks)]
        self.refine = nn.Sequential(*blocks)
        self.expand = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.refine(x)
        x = self.expand(x)
        return F.normalize(x, p=2, dim=-1)

class SiameseHead(nn.Module):
    def __init__(self, in_dim: int = 4096, mid_dim: int = 1024, out_dim: int = 2048, n_blocks: int = 6):
        super().__init__()
        self.head = ProjectionHead(in_dim, mid_dim, out_dim, n_blocks)

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v


'''
============================================================
Loss Function (v0)
    구성:
      - CD  : benign 거리 자체를 줄이는 항 (현재 lambda_cd=0.0이면 off)
      - CDD : (pathogenic 거리) - (benign 거리) 를 키우는 마진 기반 항
      - PCC : 그룹 내에서 var_len과 distance의 정렬/상관을 강제
              * z-score MSE (corr term)
              * neighbor monotonicity
              * Spearman-ish pairwise rank
    특징:
      - PCC는 group 단위로 계산 후, group size 비례 가중 평균
============================================================
'''
class MetricsLoss(nn.Module):
    def __init__(
        self,
        margin: float = 2.0,
        lambda_cd: float = 0.0,
        lambda_cdd: float = 1.0,
        lambda_pcc: float = 1.0,
        k_margin: float = 0.02,
    ):
        super().__init__()
        self.margin = margin
        self.lambda_cd = lambda_cd
        self.lambda_cdd = lambda_cdd
        self.lambda_pcc = lambda_pcc
        self.k_margin = k_margin

    def forward(
        self,
        z_r: torch.Tensor,
        z_v: torch.Tensor,
        labels: torch.Tensor,
        groups: torch.Tensor,
        var_lens: torch.Tensor,
    ):
        d = cosine_distance(z_r, z_v)  # [B]

        # CD (lambda_cd=0.0이면 off)
        if self.lambda_cd > 0:
            mask_b = (labels == 0)
            if mask_b.sum() > 0:
                l_cd = d[mask_b].mean()
            else:
                l_cd = torch.tensor(0.0, device=d.device)
        else:
            l_cd = torch.tensor(0.0, device=d.device)

        # CDD
        d_b = d[labels == 0]
        d_p = d[labels == 1]
        if d_b.numel() > 0 and d_p.numel() > 0:
            l_cdd = F.relu(self.margin + d_b.mean() - d_p.mean())
        else:
            l_cdd = torch.tensor(0.0, device=d.device)

        # PCC (group-wise)
        group_losses = []
        group_sizes  = []

        unique_groups = torch.unique(groups)

        for g in unique_groups:
            idx_t = (groups == g).nonzero(as_tuple=True)[0]
            n_g = idx_t.numel()
            if n_g < 2:
                continue

            vlen = var_lens[idx_t].float()  # [n_g]
            dist_g = d[idx_t]               # [n_g]

            # 1) z-score MSE (corr term)
            if vlen.std() > 0 and dist_g.std() > 0:
                vlen_z = (vlen - vlen.mean()) / (vlen.std() + 1e-6)
                dist_z = (dist_g - dist_g.mean()) / (dist_g.std() + 1e-6)
                corr_loss = (vlen_z - dist_z).pow(2).mean()
            else:
                corr_loss = torch.tensor(0.0, device=d.device)

            # 길이 기준 정렬
            order = torch.argsort(vlen)
            v_sorted = vlen[order]
            d_sorted = dist_g[order]

            # 2) neighbor monotonicity: len 증가 → dist 증가
            if d_sorted.numel() > 1:
                neigh_viol = F.relu(d_sorted[:-1] - d_sorted[1:] + self.k_margin).mean()
            else:
                neigh_viol = torch.tensor(0.0, device=d.device)

            # 3) Spearman-ish pairwise rank: 모든 (i,j) 페어
            dv = v_sorted.unsqueeze(0) - v_sorted.unsqueeze(1)  # [n_g, n_g]
            dd = d_sorted.unsqueeze(0) - d_sorted.unsqueeze(1)  # [n_g, n_g]

            mask = dv > 0   # len_j > len_i
            if mask.any():
                pair_viol = F.relu(self.k_margin - dd[mask])
                rank_loss = pair_viol.mean()
            else:
                rank_loss = torch.tensor(0.0, device=d.device)

            group_loss = corr_loss + neigh_viol + rank_loss
            group_losses.append(group_loss)
            group_sizes.append(float(n_g))

        if group_losses:
            group_losses_t = torch.stack(group_losses)                  # [G]
            sizes_t        = torch.tensor(group_sizes, device=d.device) # [G]
            weights        = sizes_t / sizes_t.sum()
            l_pcc = (weights * group_losses_t).sum()
        else:
            l_pcc = torch.tensor(0.0, device=d.device)

        total = (
            self.lambda_cd * l_cd +
            self.lambda_cdd * l_cdd +
            self.lambda_pcc * l_pcc
        )

        return total, {"CDD": l_cdd.item(), "PCC_loss": l_pcc.item()}, d


'''
============================================================
Evaluation (CD / CDD / PCC)
    - CD  : mean cosine distance
    - CDD : (patho mean dist) - (benign mean dist)
    - PCC : group별 Pearson corr(var_len, dist) 평균
============================================================
'''
@torch.no_grad()
def evaluate_epoch(net: nn.Module, dl: DataLoader, device: str, loss_fn: MetricsLoss) -> Dict[str, Any]:
    net.eval()
    total_loss, n_loss = 0.0, 0
    all_d, all_label, all_group, all_var_len = [], [], [], []

    for ref_e, var_e, labels, groups, var_lens in tqdm(dl, desc="Eval", leave=False):
        ref_t = torch.tensor(ref_e, dtype=torch.float32, device=device)
        var_t = torch.tensor(var_e, dtype=torch.float32, device=device)
        lbl_t = torch.tensor(labels, dtype=torch.long, device=device)
        grp_t = torch.tensor(groups, dtype=torch.long, device=device)
        vlen_t = torch.tensor(var_lens, dtype=torch.float32, device=device)

        z_r, z_v = net(ref_t, var_t)
        loss, _, d = loss_fn(z_r, z_v, lbl_t, grp_t, vlen_t)

        total_loss += float(loss)
        n_loss += 1
        all_d.append(d.cpu())
        all_label.append(labels)
        all_group.append(groups)
        all_var_len.append(var_lens)

    if not all_d:
        return {"loss": 0.0, "CD": 0.0, "CD_Benign": 0.0, "CD_Patho": 0.0, "CDD": 0.0, "PCC": 0.0}

    d_all = torch.cat(all_d).numpy()
    label_all = np.concatenate(all_label)
    group_all = np.concatenate(all_group)
    vlen_all = np.concatenate(all_var_len)

    cd = float(d_all.mean())
    mask_b, mask_p = (label_all == 0), (label_all == 1)
    db = float(d_all[mask_b].mean()) if mask_b.any() else cd
    dp = float(d_all[mask_p].mean()) if mask_p.any() else cd
    cdd = dp - db

    pcc_list = []
    for g in np.unique(group_all):
        idxs = np.where(group_all == g)[0]
        if len(idxs) < 2:
            continue
        vx = vlen_all[idxs]
        vy = d_all[idxs]
        if np.std(vx) == 0 or np.std(vy) == 0:
            continue
        r = np.corrcoef(vx, vy)[0, 1]
        if not np.isnan(r):
            pcc_list.append(r)
    pcc = float(np.mean(pcc_list)) if pcc_list else 0.0

    return {
        "loss": float(total_loss / max(1, n_loss)),
        "CD": cd,
        "CD_Benign": db,
        "CD_Patho": dp,
        "CDD": cdd,
        "PCC": pcc,
    }


'''
============================================================
Config
============================================================
'''
@dataclass
class TrainingConfig:
    emb_path: str
    out_dir: str
    batch_size: int = 2048
    epochs: int = 100
    lr: float = 1e-3


'''
============================================================
Greedy Soup over epochs (v0, val 기반)
    - 각 epoch ckpt를 점수(CD+CDD+PCC) 내림차순으로 정렬
    - greedy하게 평균(soup)하여 val score 개선 시에만 채택
    - 최종 souped state_dict 저장: head_best_souped_v0.pt
============================================================
'''
def greedy_soup_epochs(
    cfg: TrainingConfig,
    epoch_infos: List[Dict[str, Any]],
    dl_val: DataLoader,
    in_dim: int,
    device: str = "cuda",
):
    print(f"\n🥣 [v0 Epoch Soup] Starting Greedy Soup over {len(epoch_infos)} epochs...")

    eval_loss_fn = MetricsLoss(margin=2.0, lambda_cd=0.0, lambda_cdd=1.0, lambda_pcc=1.0)
    net = SiameseHead(in_dim=in_dim, mid_dim=1024, out_dim=2048, n_blocks=6).to(device)

    # CD + CDD + PCC 기준으로 정렬 (내림차순)
    epoch_infos_sorted = sorted(epoch_infos, key=lambda x: x["score"], reverse=True)

    best_info = epoch_infos_sorted[0]
    best_score = best_info["score"]
    soup_state = torch.load(best_info["path"], map_location=device)
    num_ingredients = 1

    print(
        f"[v0 Epoch Soup] Initial Best: epoch {best_info['epoch']} | "
        f"score={best_score:.4f} (CD={best_info['CD']:.4f}, "
        f"CDD={best_info['CDD']:.4f}, PCC={best_info['PCC']:.4f})"
    )

    for info in epoch_infos_sorted[1:]:
        print(
            f"\n[v0 Epoch Soup] Try add epoch {info['epoch']} "
            f"(single score={info['score']:.4f})"
        )
        cand_state = torch.load(info["path"], map_location=device)

        temp_state = {}
        for k in soup_state.keys():
            temp_state[k] = (soup_state[k] * num_ingredients + cand_state[k]) / (num_ingredients + 1)

        net.load_state_dict(temp_state)
        met = evaluate_epoch(net, dl_val, device, eval_loss_fn)
        new_score = met["CD"] + met["CDD"] + met["PCC"]

        print(
            f"   -> Mixed score={new_score:.4f} "
            f"(CD={met['CD']:.4f}, CDD={met['CDD']:.4f}, PCC={met['PCC']:.4f})"
        )

        if new_score > best_score:
            print(
                f"   [Add] Improved! {best_score:.4f} -> {new_score:.4f} "
                f"(ingredients {num_ingredients} -> {num_ingredients+1})"
            )
            best_score = new_score
            soup_state = temp_state
            num_ingredients += 1
        else:
            print(f"   [Skip] No improvement over best score {best_score:.4f}")

    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    soup_path = seed_dir / "head_best_souped_v0.pt"
    torch.save(soup_state, soup_path)
    print(f"\n [v0 Epoch Soup] Final souped model saved: {soup_path}")
    print(f"   Final Val score (CD+CDD+PCC): {best_score:.4f}, ingredients={num_ingredients}")

    final_val = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    with open(seed_dir / "v0_epoch_soup_val_summary.json", "w") as f:
        json.dump({"val": final_val, "score": best_score}, f, indent=2)

    return soup_path, final_val


'''
============================================================
main (v0)
    - seed=2025, epochs=100
    - train/val split만 사용
    - 매 epoch ckpt 저장 + best single 저장
    - 마지막에 greedy epoch soup 수행
============================================================
'''
def main():
    cfg = TrainingConfig(
        emb_path=str(EMB_PATH),
        out_dir=str(OUTPUT_DIR),
        batch_size=2048,
        epochs=100,
        lr=1e-3,
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[System] Device: {device}")

    print(f"[Data] Loading {cfg.emb_path} ...")
    full_data = np.load(cfg.emb_path, allow_pickle=True)

    # full emb 기준: split=0(train), 1(val)만 있다고 가정
    ds_train = SharedEmbeddingDataset(full_data, 0, group_bucket_size=100)
    ds_val   = SharedEmbeddingDataset(full_data, 1, group_bucket_size=100)

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_emb
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_emb
    )

    in_dim = ds_train.H
    net = SiameseHead(in_dim=in_dim, mid_dim=1024, out_dim=2048, n_blocks=6).to(device)

    loss_fn = MetricsLoss(margin=2.0, lambda_cd=0.0, lambda_cdd=1.0, lambda_pcc=1.0)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    best_single_score = float("-inf")
    best_single_epoch = None
    epoch_infos: List[Dict[str, Any]] = []

    print(
        f"\n Training Start (v0 full, seed={SEED}, "
        f"epochs={cfg.epochs}, batch={cfg.batch_size}, ResBlocks=6)"
    )

    for epoch in range(1, cfg.epochs + 1):
        # λ_pcc 스케줄: 1–10:0.0 / 11–20:0.5 / 21+:1.0
        if epoch <= 10:
            loss_fn.lambda_pcc = 0.0
        elif epoch <= 20:
            loss_fn.lambda_pcc = 0.5
        else:
            loss_fn.lambda_pcc = 1.0

        net.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(dl_train, desc=f"[v0 full Seed {SEED}] Ep {epoch}", leave=False)

        for ref_e, var_e, labels, groups, var_lens in pbar:
            ref_t = torch.tensor(ref_e, dtype=torch.float32, device=device)
            var_t = torch.tensor(var_e, dtype=torch.float32, device=device)
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)
            grp_t = torch.tensor(groups, dtype=torch.long, device=device)
            vlen_t = torch.tensor(var_lens, dtype=torch.float32, device=device)

            opt.zero_grad()
            z_r, z_v = net(ref_t, var_t)
            loss, parts, _ = loss_fn(z_r, z_v, lbl_t, grp_t, vlen_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            run_loss += float(loss)
            n += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cdd=f"{parts['CDD']:.3f}",
                lambda_pcc=f"{loss_fn.lambda_pcc:.1f}",
            )

        scheduler.step()

        # Validation
        val_met = evaluate_epoch(net, dl_val, device, loss_fn)
        score = val_met["CD"] + val_met["CDD"] + val_met["PCC"]

        ckpt_path = seed_dir / f"head_ep{epoch:03d}.pt"
        torch.save(net.state_dict(), ckpt_path)
        epoch_infos.append({
            "epoch": epoch,
            "path": str(ckpt_path),
            "score": float(score),
            "CD": float(val_met["CD"]),
            "CDD": float(val_met["CDD"]),
            "PCC": float(val_met["PCC"]),
        })

        improved = False
        if score > best_single_score:
            best_single_score = score
            best_single_epoch = epoch
            improved = True
            torch.save(net.state_dict(), seed_dir / "head_best_single.pt")
            with open(seed_dir / "v0_full_best_single_val_summary.json", "w") as f:
                json.dump(
                    {"epoch": epoch, "metrics": val_met, "score": best_single_score},
                    f,
                    indent=2,
                )

        print(
            f"[v0 full Seed {SEED}] Ep {epoch:02d} | "
            f"TrainLoss: {run_loss/n:.4f} | "
            f"ValLoss: {val_met['loss']:.4f} | "
            f"CD: {val_met['CD']:.4f} | "
            f"CDD: {val_met['CDD']:.4f} | "
            f"PCC: {val_met['PCC']:.4f} | "
            f"score(CD+CDD+PCC): {score:.4f} "
            f"(Best single: {best_single_score:.4f} @ep{best_single_epoch}{' *' if improved else ''}) | "
            f"λ_pcc(train)={loss_fn.lambda_pcc:.1f}"
        )

    with open(seed_dir / "v0_full_epoch_val_metrics.json", "w") as f:
        json.dump(epoch_infos, f, indent=2)

    print(f"\n v0 full training finished. Best single epoch = {best_single_epoch} "
          f"(score={best_single_score:.4f})")

    print("\n[Val] Evaluating v0 full best single-epoch model on Val Set...")
    best_single_path = seed_dir / "head_best_single.pt"
    net.load_state_dict(torch.load(best_single_path, map_location=device))
    eval_loss_for_val = MetricsLoss(margin=2.0, lambda_cd=0.0, lambda_cdd=1.0, lambda_pcc=1.0)
    best_single_val = evaluate_epoch(net, dl_val, device, eval_loss_for_val)
    best_single_score_val = best_single_val["CD"] + best_single_val["CDD"] + best_single_val["PCC"]

    print("=" * 60)
    print(f" v0 FULL BEST SINGLE-EPOCH VAL RESULT (score = CD + CDD + PCC)")
    print(f"   score: {best_single_score_val:.4f}")
    print(f"   CD   : {best_single_val['CD']:.4f} "
          f"(B:{best_single_val['CD_Benign']:.4f} / P:{best_single_val['CD_Patho']:.4f})")
    print(f"   CDD  : {best_single_val['CDD']:.4f}")
    print(f"   PCC  : {best_single_val['PCC']:.4f}")
    print("=" * 60)

    with open(seed_dir / "v0_full_best_single_val_final_summary.json", "w") as f:
        json.dump({"val": best_single_val, "score": best_single_score_val}, f, indent=2)

    # epoch greedy soup (Val 기반)
    greedy_soup_epochs(
        cfg=cfg,
        epoch_infos=epoch_infos,
        dl_val=dl_val,
        in_dim=in_dim,
        device=device,
    )


if __name__ == "__main__":
    main()
