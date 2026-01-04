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
[설정]
    - 학습/전처리 데이터는 for_data_and_train/local_data/ 이하 사용
    - 입력 임베딩: refvar_L1024_evo2L26_emb_full.npz (split: 0=train, 1=val)
    - 출력 체크포인트: ckpt_v2_alpha_rich/seed_2025/...
    - Teacher(v1) 체크포인트가 있으면 distillation 사용, 없으면 자동 비활성화
    - 디스크에 유지할 epoch ckpt 상위 K개만 남김 (MAX_CKPT_TO_KEEP)
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"

EMB_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_evo2L26_emb_full.npz"  # full emb, split: 0=train, 1=val
OUTPUT_DIR = PROJECT_ROOT / "ckpt_v2_alpha_rich"

TEACHER_CKPT = PROJECT_ROOT / "ckpt_v1_seed2025_100ep" / "seed_2025" / "head_best_souped_v1.pt"

SEED = 2025
MAX_CKPT_TO_KEEP = 10  # 상위 10개 epoch만 디스크에 유지

print(f"[System] Input Data: {EMB_PATH}")
print(f"[System] Output Dir: {OUTPUT_DIR}")
print(f"[System] Seed: {SEED}")
print(f"[System] Teacher (v1): {TEACHER_CKPT}")

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {EMB_PATH}")


'''
============================================================
유틸리티
    - set_seed: 재현성 확보
    - cosine_distance: L2 normalize된 벡터 가정 하 cosine distance 계산
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
    - npz 내부 split으로 train/val 선택
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
Teacher(v1)용 기본 블록 (alpha 없음)
    - teacher 구조와 맞춰야 하므로 "alpha 없는" projection/residual 사용
============================================================
'''
class ProjectionBlockV1(nn.Module):
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


class ResidualBlockV1(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
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


'''
============================================================
Student(v2)용 블록 (alpha + rich MLP)
    - ProjectionBlockV2: y = W_skip(x) + alpha * f(x)
    - ResidualBlockV2:   y = x + alpha * f(x)
    - RichResidualBlock2048: 2-layer MLP residual (alpha 포함)
============================================================
'''
class ProjectionBlockV2(nn.Module):
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

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = self.ln(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return skip + self.alpha * h


class ResidualBlockV2(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return x + self.alpha * h


class RichResidualBlock2048(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + self.alpha * h


'''
============================================================
Teacher: v1 Head (ProjectionBlockV1 + ResidualBlockV1×8)
    - distillation target로 사용
    - 체크포인트(TEACHER_CKPT)가 있을 때만 활성화
============================================================
'''
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
        assert mid2_dim == out_dim
        self.proj1 = ProjectionBlockV1(in_dim, mid1_dim, dropout=0.1)
        self.proj2 = ProjectionBlockV1(mid1_dim, mid2_dim, dropout=0.1)
        blocks = [ResidualBlockV1(dim=mid2_dim, dropout=0.1) for _ in range(n_blocks)]
        self.refine = nn.Sequential(*blocks)
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
Student: v2 Head
    - 구조:
        4096 → 3072 (proj, alpha)
        3072 Res × 4 (alpha)
        3072 → 2048 (proj, alpha)
        2048 Res × 12 (마지막 4개는 rich MLP)
    - 핵심 의도:
        * v1에서 학습된 하위(2048) 표현을 최대한 보존하면서
          alpha를 통해 점진적으로 비선형을 "켜는" 방식으로 안정화
============================================================
'''
class ProjectionHeadV2(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks_3072: int = 4,
        n_blocks_2048: int = 12,
        n_rich_2048: int = 4,
    ):
        super().__init__()
        assert mid2_dim == out_dim

        self.proj1 = ProjectionBlockV2(in_dim, mid1_dim, dropout=0.1)

        self.high = nn.Sequential(
            *[ResidualBlockV2(dim=mid1_dim, dropout=0.1) for _ in range(n_blocks_3072)]
        )

        self.proj2 = ProjectionBlockV2(mid1_dim, mid2_dim, dropout=0.1)

        blocks_low = []
        n_simple = max(0, n_blocks_2048 - n_rich_2048)
        for i in range(n_blocks_2048):
            if i < n_simple:
                blocks_low.append(ResidualBlockV2(dim=mid2_dim, dropout=0.1))
            else:
                blocks_low.append(RichResidualBlock2048(dim=mid2_dim, hidden=mid2_dim * 2, dropout=0.1))
        self.low = nn.Sequential(*blocks_low)

        self.out_ln = nn.LayerNorm(mid2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.high(x)
        x = self.proj2(x)
        x = self.low(x)
        x = self.out_ln(x)
        return F.normalize(x, p=2, dim=-1)


class SiameseHeadV2(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks_3072: int = 4,
        n_blocks_2048: int = 12,
        n_rich_2048: int = 4,
    ):
        super().__init__()
        self.head = ProjectionHeadV2(
            in_dim=in_dim,
            mid1_dim=mid1_dim,
            mid2_dim=mid2_dim,
            out_dim=out_dim,
            n_blocks_3072=n_blocks_3072,
            n_blocks_2048=n_blocks_2048,
            n_rich_2048=n_rich_2048,
        )

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v


'''
============================================================
초기화: v2를 v1로부터 warm-start
    - v1 head에서 proj1/proj2 + 2048 ResBlock(앞 8개)를 v2에 복사
    - v2의 3072 블록/추가 2048 블록은 zero-init + alpha=0 그대로 유지
    - 복사 시 alpha 파라미터는 v2 쪽(0)을 유지하도록 "alpha 제외" 정책 적용
============================================================
'''
def init_v2_from_v1(v2: SiameseHeadV2, v1: SiameseHeadV1):
    with torch.no_grad():
        v2_head = v2.head
        v1_head = v1.head

        # proj1: weight 복사 (alpha는 v2에서 0 유지)
        v2_proj1_state = v2_head.proj1.state_dict()
        v1_proj1_state = v1_head.proj1.state_dict()
        for k in v1_proj1_state:
            if k in v2_proj1_state and "alpha" not in k:
                v2_proj1_state[k] = v1_proj1_state[k]
        v2_head.proj1.load_state_dict(v2_proj1_state)

        # proj2: weight 복사 (alpha는 v2에서 0 유지)
        v2_proj2_state = v2_head.proj2.state_dict()
        v1_proj2_state = v1_head.proj2.state_dict()
        for k in v1_proj2_state:
            if k in v2_proj2_state and "alpha" not in k:
                v2_proj2_state[k] = v1_proj2_state[k]
        v2_head.proj2.load_state_dict(v2_proj2_state)

        # 2048 ResBlock: v1.refine[0..7] → v2.low의 simple block들에 순서대로 복사
        num_copy = 0
        for blk in v2_head.low:
            if isinstance(blk, ResidualBlockV2) and num_copy < len(v1_head.refine):
                v2_blk_state = blk.state_dict()
                v1_blk_state = v1_head.refine[num_copy].state_dict()
                for k in v1_blk_state:
                    if k in v2_blk_state and "alpha" not in k:
                        v2_blk_state[k] = v1_blk_state[k]
                blk.load_state_dict(v2_blk_state)
                num_copy += 1

        print(f"[Init] v2 head initialized from v1 for proj1/proj2 + {num_copy} low(simple) blocks.")


'''
============================================================
Loss Function (CDD + PCC)
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

        # PCC
        group_losses = []
        group_sizes  = []
        unique_groups = torch.unique(groups)

        for g in unique_groups:
            idx_t = (groups == g).nonzero(as_tuple=True)[0]
            n_g = idx_t.numel()
            if n_g < 2:
                continue

            vlen = var_lens[idx_t].float()
            dist_g = d[idx_t]

            if vlen.std() > 0 and dist_g.std() > 0:
                vlen_z = (vlen - vlen.mean()) / (vlen.std() + 1e-6)
                dist_z = (dist_g - dist_g.mean()) / (dist_g.std() + 1e-6)
                corr_loss = (vlen_z - dist_z).pow(2).mean()
            else:
                corr_loss = torch.tensor(0.0, device=d.device)

            order = torch.argsort(vlen)
            v_sorted = vlen[order]
            d_sorted = dist_g[order]

            if d_sorted.numel() > 1:
                neigh_viol = F.relu(
                    d_sorted[:-1] - d_sorted[1:] + self.k_margin
                ).mean()
            else:
                neigh_viol = torch.tensor(0.0, device=d.device)

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
            sizes_t        = torch.tensor(group_sizes, device=d.device)
            weights        = sizes_t / sizes_t.sum()
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
    epochs: int = 100   # 필요하면 150으로 올려도 됨
    lr: float = 3e-4


'''
============================================================
Greedy Soup (alpha-tuned mixing)
    - epoch_infos: score 내림차순 정렬
    - 상위 30%만 후보로 사용
    - soup_state(현재)와 cand_state(후보)를 alpha grid로 섞어서
      val score(CD+CDD+PCC)가 개선되면 ingredient로 채택
    - 최종 souped state 저장: head_best_souped_v2.pt
============================================================
'''
def greedy_soup_epochs(
    cfg: TrainingConfig,
    epoch_infos: List[Dict[str, Any]],
    dl_val: DataLoader,
    in_dim: int,
    device: str = "cuda",
):
    print(f"\n[v2 Epoch Soup] Starting Greedy Soup over {len(epoch_infos)} epochs...")

    eval_loss_fn = MetricsLoss(margin=2.0, k_margin=0.02)
    net = SiameseHeadV2(in_dim=in_dim).to(device)

    if len(epoch_infos) == 0:
        print("[v2 Epoch Soup] No epoch_infos provided. Skipping soup.")
        return None, None

    epoch_infos_sorted = sorted(epoch_infos, key=lambda x: x["score"], reverse=True)

    total_epochs = len(epoch_infos_sorted)
    top_k = max(1, int(total_epochs * 0.3))
    top_epoch_infos = epoch_infos_sorted[:top_k]

    print(f"[v2 Epoch Soup] Using top {top_k} / {total_epochs} epochs for soup")

    # 초기 soup = 최고 single 모델
    best_info = top_epoch_infos[0]
    soup_state = torch.load(best_info["path"], map_location="cpu")
    net.load_state_dict(soup_state)
    met0 = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    best_score = met0["CD"] + met0["CDD"] + met0["PCC"]
    num_ingredients = 1

    print(
        f"[v2 Epoch Soup] Initial soup from epoch {best_info['epoch']} | "
        f"score={best_score:.4f} (CD={met0['CD']:.4f}, "
        f"CDD={met0['CDD']:.4f}, PCC={met0['PCC']:.4f})"
    )

    alphas = [0.25, 0.5, 0.75, 1.0]

    for info in top_epoch_infos[1:]:
        print(f"\n[v2 Epoch Soup] Try add epoch {info['epoch']} (single score={info['score']:.4f})")
        cand_state = torch.load(info["path"], map_location="cpu")

        best_mix_local_score = best_score
        best_mix_local_state = None
        best_mix_alpha = None

        for alpha in alphas:
            mix_state = {}
            for k in soup_state.keys():
                mix_state[k] = (1 - alpha) * soup_state[k] + alpha * cand_state[k]

            net.load_state_dict(mix_state)
            met = evaluate_epoch(net, dl_val, device, eval_loss_fn)
            mix_score = met["CD"] + met["CDD"] + met["PCC"]

            print(
                f"   alpha_mix={alpha:.2f} -> score={mix_score:.4f} "
                f"(CD={met['CD']:.4f}, CDD={met['CDD']:.4f}, PCC={met['PCC']:.4f})"
            )

            if mix_score > best_mix_local_score:
                best_mix_local_score = mix_score
                best_mix_local_state = {k: v.clone() for k, v in mix_state.items()}
                best_mix_alpha = alpha

        if best_mix_local_state is not None:
            print(
                f"   [Add] Improved soup with alpha_mix={best_mix_alpha:.2f}: "
                f"{best_score:.4f} -> {best_mix_local_score:.4f} "
                f"(ingredients {num_ingredients} -> {num_ingredients+1})"
            )
            soup_state = best_mix_local_state
            best_score = best_mix_local_score
            num_ingredients += 1
        else:
            print(f"   [Skip] No alpha_mix improved over current best score {best_score:.4f}")

    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    soup_path = seed_dir / "head_best_souped_v2.pt"
    torch.save(soup_state, soup_path)
    print(f"\n[v2 Epoch Soup] Final souped model saved: {soup_path}")
    print(f"[v2 Epoch Soup] Final Val score (CD+CDD+PCC): {best_score:.4f}, ingredients={num_ingredients}")

    net.load_state_dict(soup_state)
    final_val = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    with open(seed_dir / "v2_epoch_soup_val_summary.json", "w") as f:
        json.dump({"val": final_val, "score": best_score}, f, indent=2)

    return soup_path, final_val


'''
============================================================
main
    - Student(v2) 학습
    - Teacher(v1) 존재 시 distillation 사용 (초기 몇 epoch)
    - epoch ckpt 저장 후, 상위 MAX_CKPT_TO_KEEP개만 디스크에 유지
    - 마지막에 greedy soup는 "남아있는 상위 ckpt 목록(best_ckpts)"만 사용
============================================================
'''
def main():
    cfg = TrainingConfig(
        emb_path=str(EMB_PATH),
        out_dir=str(OUTPUT_DIR),
        batch_size=4096,
        epochs=100,   # 더 길게 돌리려면 150으로 변경
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

    # Student: v2
    net = SiameseHeadV2(in_dim=in_dim).to(device)

    # Teacher: v1
    teacher = SiameseHeadV1(in_dim=in_dim).to(device)
    use_teacher = TEACHER_CKPT.exists()
    if use_teacher:
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
        print("[Teacher] Loaded v1 head from", TEACHER_CKPT)
        init_v2_from_v1(net, teacher)
    else:
        print("[Teacher] WARNING: teacher (v1) checkpoint not found, distillation disabled.")

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
    epoch_infos: List[Dict[str, Any]] = []
    best_ckpts: List[Dict[str, Any]] = []

    print(
        f"\n[Train] Start (v2, seed={SEED}, epochs={cfg.epochs}, batch={cfg.batch_size}, "
        f"ResBlocks: 3072x4 + 2048x12 (last 4 rich))"
    )

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(dl_train, desc=f"[v2 Seed {SEED}] Ep {epoch}", leave=False)

        for ref_e, var_e, labels, groups, var_lens in pbar:
            ref_t = torch.tensor(ref_e, dtype=torch.float32, device=device)
            var_t = torch.tensor(var_e, dtype=torch.float32, device=device)
            lbl_t = torch.tensor(labels, dtype=torch.long, device=device)
            grp_t = torch.tensor(groups, dtype=torch.long, device=device)
            vlen_t = torch.tensor(var_lens, dtype=torch.float32, device=device)

            opt.zero_grad()
            z_r, z_v = net(ref_t, var_t)
            l_cdd, l_pcc, d = loss_fn(z_r, z_v, lbl_t, grp_t, vlen_t)

            cd = d.mean()
            l_cdreg = F.relu(0.2 - cd)

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
                # Phase 1: loss = CDD + 0.5*PCC + CD_reg + 0.5*dist
                loss = l_cdd + 0.5 * l_pcc + l_cdreg + (0.5 * l_dist if use_teacher else 0.0)
                lambda_pcc = 0.5
                lambda_dist = 0.5 if use_teacher else 0.0

            else:
                # Phase 2: loss = CDD + PCC + CD_reg (dist=0)
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
                lam_pcc=f"{lambda_pcc:.2f}",
                lam_dist=f"{lambda_dist:.2f}",
            )

        scheduler.step()

        val_met = evaluate_epoch(net, dl_val, device, loss_fn)
        score = val_met["CD"] + val_met["CDD"] + val_met["PCC"]

        # ckpt 저장 + 상위 K개만 유지
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
        while len(best_ckpts) > MAX_CKPT_TO_KEEP:
            worst = best_ckpts.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])

        improved = False
        if score > best_single_score:
            best_single_score = score
            best_single_epoch = epoch
            improved = True
            torch.save(net.state_dict(), seed_dir / "head_best_single.pt")
            with open(seed_dir / "v2_best_single_val_summary.json", "w") as f:
                json.dump(
                    {"epoch": epoch, "metrics": val_met, "score": best_single_score},
                    f,
                    indent=2,
                )

        print(
            f"[v2 Seed {SEED}] Ep {epoch:02d} | "
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
    with open(seed_dir / "v2_epoch_val_metrics.json", "w") as f:
        json.dump(epoch_infos, f, indent=2)

    print(
        f"\n[Done] v2 training finished. Best single epoch = {best_single_epoch} "
        f"(score={best_single_score:.4f})"
    )

    # best single-epoch 모델로 Val 재평가
    print("\n[Val] Evaluating v2 best single-epoch model on Val Set...")
    best_single_path = seed_dir / "head_best_single.pt"
    net.load_state_dict(torch.load(best_single_path, map_location=device))
    eval_loss_for_val = MetricsLoss(margin=2.0, k_margin=0.02)
    best_single_val = evaluate_epoch(net, dl_val, device, eval_loss_for_val)
    with open(seed_dir / "v2_best_single_val_final_summary.json", "w") as f:
        json.dump({"val": best_single_val, "score": best_single_score}, f, indent=2)

    # epoch greedy soup: 실제로 남아 있는 상위 K개만 사용
    greedy_soup_epochs(
        cfg=cfg,
        epoch_infos=best_ckpts,
        dl_val=dl_val,
        in_dim=in_dim,
        device=device,
    )


if __name__ == "__main__":
    main()
