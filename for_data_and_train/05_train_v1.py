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
[설정] (v1)
    - 학습/전처리 데이터는 for_data_and_train/local_data/ 이하 사용
    - 입력 임베딩: refvar_L1024_evo2L26_emb_full.npz (split: 0=train, 1=val)
    - 출력 체크포인트: ckpt_v1_seed2025_100ep/seed_2025/...
    - Teacher(v0) 체크포인트가 있으면 distillation 사용, 없으면 자동 비활성화
    - 디스크에 유지할 epoch ckpt 상위 K개만 남김 (MAX_CKPT_TO_KEEP)
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"

EMB_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_evo2L26_emb_full.npz"
OUTPUT_DIR = PROJECT_ROOT / "ckpt_v1_seed2025_100ep"

TEACHER_CKPT = PROJECT_ROOT / "ckpt_v0" / "seed_2025" / "head_best_souped_v0.pt"

SEED = 2025
MAX_CKPT_TO_KEEP = 10

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

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(dim=-1)


'''
============================================================
Dataset
    - npz 내부에서 split으로 train/val 선택
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

        self.group = raw_group // group_bucket_size
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
Student: v1 Head (4096 -> 3072 -> 2048, projection residual + ResBlock(2048) x 8)
    - ProjectionBlock:
        y = W_skip(x) + f(x)
        f(x) = LN(x) -> Linear(d_in->d_out) -> GELU -> Dropout
        Linear는 zero-init (초기에는 y ≈ W_skip(x))
    - ResidualBlock2048:
        y = x + f(x)
        f(x) = LN(x) -> Linear(2048->2048) -> GELU -> Dropout
        Linear zero-init (초기에는 y ≈ x)
============================================================
'''
class ProjectionBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out, bias=False)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = self.ln(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return skip + h

class ResidualBlock2048(nn.Module):
    def __init__(self, dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return x + h

class ProjectionHeadV1(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks: int = 8,
    ):
        super().__init__()
        assert mid2_dim == out_dim, "v1: mid2_dim must equal out_dim (2048)."

        # 4096 → 3072 projection residual
        self.proj1 = ProjectionBlock(d_in=in_dim, d_out=mid1_dim, dropout=0.1)
        # 3072 → 2048 projection residual
        self.proj2 = ProjectionBlock(d_in=mid1_dim, d_out=mid2_dim, dropout=0.1)

        # 2048 → 2048 residual blocks
        blocks = [ResidualBlock2048(dim=mid2_dim, dropout=0.1) for _ in range(n_blocks)]
        self.refine = nn.Sequential(*blocks)
        # 최종 LN + L2
        self.out_ln = nn.LayerNorm(mid2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.proj2(x)
        x = self.refine(x)
        x = self.out_ln(x)
        return F.normalize(x, p=2, dim=-1)
    

class SiameseHeadV1(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks: int = 8,
    ):
        super().__init__()
        self.head = ProjectionHeadV1(in_dim, mid1_dim, mid2_dim, out_dim, n_blocks)

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v

'''
============================================================
Teacher: v0 Head (4096 -> 1024 -> ResBlock x 6 -> 2048)
    - 기존 v6 아키텍처를 "v0"로 명명만 변경
    - teacher ckpt 로드 시 distillation에 사용
============================================================
'''
class ResidualBlockV0(nn.Module):
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

class ProjectionHeadV0(nn.Module):
    def __init__(self, in_dim: int = 4096, mid_dim: int = 1024, out_dim: int = 2048, n_blocks: int = 6):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        blocks = [ResidualBlockV0(mid_dim, dropout=0.1) for _ in range(n_blocks)]
        self.refine = nn.Sequential(*blocks)
        self.expand = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = self.refine(x)
        x = self.expand(x)
        return F.normalize(x, p=2, dim=-1)

class SiameseHeadV0(nn.Module):
    def __init__(self, in_dim: int = 4096, mid_dim: int = 1024, out_dim: int = 2048, n_blocks: int = 6):
        super().__init__()
        self.head = ProjectionHeadV0(in_dim, mid_dim, out_dim, n_blocks)

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v

'''
============================================================
MetricsLoss (CDD + PCC)
    - CDD: pathogenic 거리 - benign 거리 차이를 margin 기반으로 키움
    - PCC: group 단위로 var_len과 distance의 정렬/상관을 강제
============================================================
'''
class MetricsLoss(nn.Module):
    def __init__(self, margin: float = 2.0, k_margin: float = 0.02):
        super().__init__()
        self.margin = margin
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

        # CDD
        d_b = d[labels == 0]
        d_p = d[labels == 1]
        if d_b.numel() > 0 and d_p.numel() > 0:
            l_cdd = F.relu(self.margin + d_b.mean() - d_p.mean())
        else:
            l_cdd = torch.tensor(0.0, device=d.device)

        # PCC (group-wise)
        group_losses = []
        group_sizes = []
        unique_groups = torch.unique(groups)

        for g in unique_groups:
            idx_t = (groups == g).nonzero(as_tuple=True)[0]
            n_g = idx_t.numel()
            if n_g < 2:
                continue

            vlen = var_lens[idx_t].float()
            dist_g = d[idx_t]

            # z-score MSE
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

            # neighbor monotonicity
            if d_sorted.numel() > 1:
                neigh_viol = F.relu(d_sorted[:-1] - d_sorted[1:] + self.k_margin).mean()
            else:
                neigh_viol = torch.tensor(0.0, device=d.device)

            # Spearman-ish pairwise rank
            dv = v_sorted.unsqueeze(0) - v_sorted.unsqueeze(1)
            dd = d_sorted.unsqueeze(0) - d_sorted.unsqueeze(1)
            mask = dv > 0
            if mask.any():
                pair_viol = F.relu(self.k_margin - dd[mask])
                rank_loss = pair_viol.mean()
            else:
                rank_loss = torch.tensor(0.0, device=d.device)

            group_loss = corr_loss + neigh_viol + rank_loss
            group_losses.append(group_loss)
            group_sizes.append(float(n_g))

        if group_losses:
            group_losses_t = torch.stack(group_losses)
            sizes_t = torch.tensor(group_sizes, device=d.device)
            weights = sizes_t / sizes_t.sum()
            l_pcc = (weights * group_losses_t).sum()
        else:
            l_pcc = torch.tensor(0.0, device=d.device)

        return l_cdd, l_pcc, d

'''
============================================================
Evaluation (CD / CDD / PCC)
    - loss는 (CDD + PCC)로 보고
    - score는 (CD + CDD + PCC)로 계산
============================================================
'''
@torch.no_grad()
def evaluate_epoch(
    net: nn.Module,
    dl: DataLoader,
    device: str,
    loss_fn: MetricsLoss,
) -> Dict[str, Any]:
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
        l_cdd, l_pcc, d = loss_fn(z_r, z_v, lbl_t, grp_t, vlen_t)

        loss_batch = l_cdd + l_pcc
        total_loss += loss_batch.item()
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
    batch_size: int = 4096
    epochs: int = 100
    lr: float = 3e-4

'''
============================================================
Greedy Soup (상위 30% epoch만 대상)
    - epoch_infos를 score 내림차순 정렬
    - 상위 30%만 후보로 사용
    - greedy averaging으로 val score 개선 시에만 ingredient 추가
    - 최종 souped state 저장: head_best_souped_v1.pt
============================================================
'''
def greedy_soup_epochs(
    cfg: TrainingConfig,
    epoch_infos: List[Dict[str, Any]],
    dl_val: DataLoader,
    in_dim: int,
    device: str = "cuda",
):
    print(f"\n[v1 Epoch Soup] Starting Greedy Soup over {len(epoch_infos)} epochs...")

    eval_loss_fn = MetricsLoss(margin=2.0, k_margin=0.02)
    net = SiameseHeadV1(in_dim=in_dim, mid1_dim=3072, mid2_dim=2048, out_dim=2048, n_blocks=8).to(device)

    if len(epoch_infos) == 0:
        print("[v1 Epoch Soup] No epoch_infos provided. Skipping soup.")
        return None, None

    epoch_infos_sorted = sorted(epoch_infos, key=lambda x: x["score"], reverse=True)
    total_epochs = len(epoch_infos_sorted)
    top_k = max(1, int(total_epochs * 0.3))
    top_epoch_infos = epoch_infos_sorted[:top_k]

    print(f"[v1 Epoch Soup] Using top {top_k} / {total_epochs} epochs for soup")

    best_info = top_epoch_infos[0]
    best_score = best_info["score"]
    soup_state = torch.load(best_info["path"], map_location=device)
    num_ingredients = 1

    print(
        f"[v1 Epoch Soup] Initial Best: epoch {best_info['epoch']} | "
        f"score={best_score:.4f} (CD={best_info['CD']:.4f}, "
        f"CDD={best_info['CDD']:.4f}, PCC={best_info['PCC']:.4f})"
    )

    for info in top_epoch_infos[1:]:
        print(f"\n[v1 Epoch Soup] Try add epoch {info['epoch']} (single score={info['score']:.4f})")
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
            print(f"   [Add] Improved! {best_score:.4f} -> {new_score:.4f}")
            best_score = new_score
            soup_state = temp_state
            num_ingredients += 1
        else:
            print(f"   [Skip] No improvement over best score {best_score:.4f}")

    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    soup_path = seed_dir / "head_best_souped_v1.pt"
    torch.save(soup_state, soup_path)
    print(f"\n[v1 Epoch Soup] Final souped model saved: {soup_path}")
    print(f"[v1 Epoch Soup] Final Val score (CD+CDD+PCC): {best_score:.4f}, ingredients={num_ingredients}")

    net.load_state_dict(soup_state)
    final_val = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    with open(seed_dir / "v1_epoch_soup_val_summary.json", "w") as f:
        json.dump({"val": final_val, "score": best_score}, f, indent=2)

    return soup_path, final_val


'''
============================================================
main
    - Student(v1) 학습
    - Teacher(v0) 존재 시 distillation 사용 (초기 몇 epoch)
    - epoch ckpt 저장 후, 상위 MAX_CKPT_TO_KEEP개만 디스크에 유지
    - 마지막에 greedy soup는 "남아있는 상위 ckpt 목록(best_ckpts)"만 사용
============================================================
'''
def main():
    cfg = TrainingConfig(
        emb_path=str(EMB_PATH),
        out_dir=str(OUTPUT_DIR),
        batch_size=4096,
        epochs=100,
        lr=3e-4,
    )

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"[System] Device: {device}")
    print(f"[Data] Loading {cfg.emb_path} ...")

    full_data = np.load(cfg.emb_path, allow_pickle=True)

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
    net = SiameseHeadV1(in_dim=in_dim, mid1_dim=3072, mid2_dim=2048, out_dim=2048, n_blocks=8).to(device)

    teacher = SiameseHeadV0(in_dim=in_dim, mid_dim=1024, out_dim=2048, n_blocks=6).to(device)
    use_teacher = TEACHER_CKPT.exists()
    if use_teacher:
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
        print(f"[Teacher] Loaded teacher ckpt: {TEACHER_CKPT}")
    else:
        print(f"[Teacher] Teacher ckpt not found: {TEACHER_CKPT}")
        print("[Teacher] Distillation disabled.")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    loss_fn = MetricsLoss(margin=2.0, k_margin=0.02)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    best_single_score = float("-inf")
    best_single_epoch = None
    epoch_infos: List[Dict[str, Any]] = []   # 전체 epoch 기록용
    best_ckpts: List[Dict[str, Any]] = []    # 실제로 디스크에 남겨둘 상위 K개

    print(f"\n[Train] Start (v1, seed={SEED}, epochs={cfg.epochs}, batch={cfg.batch_size})")

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(dl_train, desc=f"[v1 Seed {SEED}] Ep {epoch}", leave=False)

        for ref_e, var_e, labels, groups, var_lens in pbar:
            ref_t = torch.tensor(ref_e, dtype=torch.float32, device=device)
            var_t = torch.tensor(var_e, dtype=torch.float32, device=device)
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)
            grp_t = torch.tensor(groups, dtype=torch.long, device=device)
            vlen_t = torch.tensor(var_lens, dtype=torch.float32, device=device)

            opt.zero_grad()

            z_r, z_v = net(ref_t, var_t)
            l_cdd, l_pcc, d = loss_fn(z_r, z_v, lbl_t, grp_t, vlen_t)

            # CD_reg: CD_mean < 0.2 일 때만 작동
            cd = d.mean()
            l_cdreg = F.relu(0.2 - cd)

            # Distillation term
            if use_teacher:
                with torch.no_grad():
                    z_r_t, z_v_t = teacher(ref_t, var_t)
                l_dist = (z_r - z_r_t.detach()).pow(2).mean() + (z_v - z_v_t.detach()).pow(2).mean()
            else:
                l_dist = torch.tensor(0.0, device=device)

            # 단계별 loss 구성 
            if use_teacher and epoch <= 5:
                # Phase 0: distillation-only (1–5)
                loss = l_dist
                lambda_pcc = 0.0
                lambda_dist = 1.0

            elif epoch <= 10:
                # Phase 1: (원래 11–20 loss를 6–10으로)
                # loss = CDD + 0.5*PCC + CD_reg + 0.5*dist
                loss = l_cdd + 0.5 * l_pcc + l_cdreg + (0.5 * l_dist if use_teacher else 0.0)
                lambda_pcc = 0.5
                lambda_dist = 0.5 if use_teacher else 0.0

            else:
                # Phase 2: 11 epoch 이후, dist=0, 나머지 1
                # loss = CDD + PCC + CD_reg
                loss = l_cdd + l_pcc + l_cdreg
                lambda_pcc = 1.0
                lambda_dist = 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            n += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cdd=f"{l_cdd.item():.3f}",
                pcc=f"{l_pcc.item():.3f}",
                cd=f"{cd.item():.3f}",
                λ_pcc=f"{lambda_pcc:.2f}",
                λ_dist=f"{lambda_dist:.2f}",
            )

        scheduler.step()

        # Validation
        val_met = evaluate_epoch(net, dl_val, device, loss_fn)
        score = val_met["CD"] + val_met["CDD"] + val_met["PCC"]

        # ckpt 저장 + 상위 10개만 유지
        ckpt_path = seed_dir / f"head_ep{epoch:03d}.pt"
        torch.save(net.state_dict(), ckpt_path)

        info = {
            "epoch": epoch,
            "path": str(ckpt_path),
            "score": float(score),
            "CD": float(val_met["CD"]),
            "CDD": float(val_met["CDD"]),
            "PCC": float(val_met["PCC"]),
        }
        epoch_infos.append(info)

        best_ckpts.append(info)
        best_ckpts.sort(key=lambda x: x["score"], reverse=True)

        # 상위 10개 초과하면 score 가장 낮은 ckpt 삭제
        while len(best_ckpts) > MAX_CKPT_TO_KEEP:
            worst = best_ckpts.pop()  # 맨 뒤 = 가장 낮은 score
            worst_path = worst["path"]
            if worst_path is not None and os.path.exists(worst_path):
                os.remove(worst_path)

        # best single-epoch 갱신
        improved = False
        if score > best_single_score:
            best_single_score = score
            best_single_epoch = epoch
            improved = True
            torch.save(net.state_dict(), seed_dir / "head_best_single.pt")
            with open(seed_dir / "v1_best_single_val_summary.json", "w") as f:
                json.dump(
                    {"epoch": epoch, "metrics": val_met, "score": best_single_score},
                    f,
                    indent=2,
                )

        print(
            f"[v1 Seed {SEED}] Ep {epoch:02d} | "
            f"TrainLoss: {run_loss/n:.4f} | "
            f"ValLoss(CDD+PCC): {val_met['loss']:.4f} | "
            f"CD: {val_met['CD']:.4f} | "
            f"CDD: {val_met['CDD']:.4f} | "
            f"PCC: {val_met['PCC']:.4f} | "
            f"score(CD+CDD+PCC): {score:.4f} "
            f"(Best single: {best_single_score:.4f} @ep{best_single_epoch}{' *' if improved else ''}) | "
            f"lambda_pcc={lambda_pcc:.2f}, lambda_dist={lambda_dist:.2f}"
        )


    # 전체 epoch 메트릭 로그 저장
    
    with open(seed_dir / "v1_epoch_val_metrics.json", "w") as f:
        json.dump(epoch_infos, f, indent=2)

    print(f"\n[Done] v1 training finished. Best single epoch = {best_single_epoch} (score={best_single_score:.4f})")

    # best single-epoch 모델로 Val 재평가
    print("\n[Val] Evaluating v1 best single-epoch model on Val Set...")
    best_single_path = seed_dir / "head_best_single.pt"
    net.load_state_dict(torch.load(best_single_path, map_location=device))
    eval_loss_for_val = MetricsLoss(margin=2.0, k_margin=0.02)
    best_single_val = evaluate_epoch(net, dl_val, device, eval_loss_for_val)
    with open(seed_dir / "v1_best_single_val_final_summary.json", "w") as f:
        json.dump({"val": best_single_val, "score": best_single_score}, f, indent=2)

    # epoch greedy soup: 실제로 남아 있는 상위 10개만 사용
    greedy_soup_epochs(
        cfg=cfg,
        epoch_infos=best_ckpts,
        dl_val=dl_val,
        in_dim=in_dim,
        device=device,
    )


if __name__ == "__main__":
    main()
