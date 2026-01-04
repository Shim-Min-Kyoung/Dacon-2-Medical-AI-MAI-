import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


'''
============================================================
[설정]
    - 학습/전처리 데이터는 for_data_and_train/local_data/ 이하 사용
    - 대회 공식 데이터는 /data (COMP_DATA_DIR)에서 읽도록 "별도 추론 스크립트"에서 처리
    - 입력 임베딩: refvar_L1024_evo2L26_emb_full.npz
    - 출력 체크포인트: ckpt_v3/seed_2025/...
    - Teacher(v2) 체크포인트가 있으면 distillation + warm-start 사용, 없으면 자동 비활성화
    - 디스크에 유지할 epoch ckpt 상위 K개만 남김 (MAX_CKPT_TO_KEEP)
    - 제출용 CSV 생성(옵션): test.csv가 있으면 souped head로 submission CSV 생성
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"
WEIGHT_DIR = PROJECT_ROOT / "weights"
CONFIG_DIR = PROJECT_ROOT / "configs"

EMB_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_evo2L26_emb_full.npz"

OUTPUT_DIR = PROJECT_ROOT / "ckpt_v3"

# Teacher(v2) souped head 경로
TEACHER_CKPT = PROJECT_ROOT / "ckpt_v2_alpha_rich" / "seed_2025" / "head_best_souped_v2.pt"

# 제출용 경로들
#  - 이 스크립트에서는 프로젝트 내부 기준으로 저장
#  - 실제 배포(대회 환경)에서는 /data를 사용하는 "별도 추론 코드"에서 처리하는 것을 권장
TEST_CSV_PATH = LOCAL_DATA_DIR / "test.csv"
MODEL_PATH = WEIGHT_DIR / "evo2_7b"
SUBMISSION_CSV_PATH = PROJECT_ROOT / "submission_v3_souped.csv"
TEST_BATCH_SIZE = 32

SEED = 2025
MAX_CKPT_TO_KEEP = 10

print(f"[System] Input Data : {EMB_PATH}")
print(f"[System] Output Dir : {OUTPUT_DIR}")
print(f"[System] Seed       : {SEED}")
print(f"[System] Teacher(v2): {TEACHER_CKPT}")

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {EMB_PATH}")


'''
============================================================
유틸
    - set_seed: 재현성 확보
    - cosine_distance: (L2 normalize된 벡터 가정) cosine distance 계산
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
Dataset (NPZ 기반 ref/var 임베딩)
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

        # coarse 그룹 버킷
        self.group = raw_group // group_bucket_size
        self.H = self.ref_emb.shape[1]

    def __len__(self):
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
RMSNorm + LayerScale + alpha (v3용)
    - RMSNorm: LayerNorm 대체(스케일 파라미터만 학습)
    - LayerScale: residual branch 출력을 채널별로 작은 값(ls_init)로 스케일
    - alpha: residual branch를 전체적으로 0에서 시작해 점진적으로 "켜는" 스칼라
============================================================
'''
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm_x * self.scale

'''
============================================================
    d_in -> d_out projection + residual:

        y = W_skip(x) + alpha * (layerscale * f(x))

    where
        f(x) = RMSNorm(x) -> Linear(d_in->d_out) -> GELU -> Dropout

    init policy:
        - Linear weight/bias: 0-init
        - alpha: 0-init
        - layerscale: small init (ls_init)
        => 초기에는 y ~= W_skip(x)에 가깝게 시작
============================================================
'''
class ProjectionBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1, ls_init: float = 1e-2):
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out, bias=False)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        self.layerscale = nn.Parameter(torch.ones(d_out) * ls_init)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        h = h * self.layerscale
        return skip + self.alpha * h

"""
============================================================
dim-d residual block:
y = x + alpha * (layerscale * f(x))
where
    f(x) = RMSNorm(x) -> Linear(dim->dim) -> GELU -> Dropout
============================================================
"""
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, ls_init: float = 1e-2):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        self.layerscale = nn.Parameter(torch.ones(dim) * ls_init)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        h = h * self.layerscale
        return x + self.alpha * h

"""
============================================================
2048-d rich residual block (2-layer MLP):

    f(x) = RMSNorm(x)
            -> Linear(2048->4096) -> GELU -> Dropout
            -> Linear(4096->2048) -> Dropout

    y = x + alpha * (layerscale * f(x))

init policy:
    - fc2: 0-init
    - alpha: 0-init
    - layerscale: small init
============================================================
"""
class RichResidualBlock2048(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 4096, dropout: float = 0.1, ls_init: float = 1e-2):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)

        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

        self.layerscale = nn.Parameter(torch.ones(dim) * ls_init)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        h = h * self.layerscale
        return x + self.alpha * h


'''
============================================================
Teacher: v2 Head (ckpt 로딩용 간단 정의)
    - distillation target으로 사용
    - v2 체크포인트 state_dict 로딩만 목적이므로 "동일 키 구조" 유지에 집중
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
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
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
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.ln(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.drop(h)
        return x + self.alpha * h


class RichResidualBlock2048V2(nn.Module):
    def __init__(self, dim: int = 2048, hidden: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + self.alpha * h


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
        self.proj1 = ProjectionBlockV2(in_dim, mid1_dim, dropout=0.1)
        self.high = nn.Sequential(*[ResidualBlockV2(mid1_dim, dropout=0.1) for _ in range(n_blocks_3072)])
        self.proj2 = ProjectionBlockV2(mid1_dim, mid2_dim, dropout=0.1)

        blocks_low = []
        n_simple = max(0, n_blocks_2048 - n_rich_2048)
        for i in range(n_blocks_2048):
            if i < n_simple:
                blocks_low.append(ResidualBlockV2(mid2_dim, dropout=0.1))
            else:
                blocks_low.append(RichResidualBlock2048V2(mid2_dim, hidden=mid2_dim * 2, dropout=0.1))
        self.low = nn.Sequential(*blocks_low)

        self.out_ln = nn.LayerNorm(mid2_dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.high(x)
        x = self.proj2(x)
        x = self.low(x)
        x = self.out_ln(x)
        return F.normalize(x, p=2, dim=-1)


class SiameseHeadV2(nn.Module):
    def __init__(self, in_dim: int = 4096):
        super().__init__()
        self.head = ProjectionHeadV2(in_dim=in_dim)

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v


'''
============================================================
Student: v3 Head
    - 구조:
        4096 -> 3072 (proj)
        3072 Res x 8
        3072 -> 2048 (proj)
        2048 Res x 16 (last 4 are rich)
        out_norm: RMSNorm
        최종: L2 normalize
============================================================
'''
class ProjectionHeadV3(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks_3072: int = 8,
        n_blocks_2048: int = 16,
        n_rich_2048: int = 4,
    ):
        super().__init__()
        assert mid2_dim == out_dim

        self.proj1 = ProjectionBlock(in_dim, mid1_dim, dropout=0.1)
        self.high = nn.Sequential(*[ResidualBlock(mid1_dim, dropout=0.1) for _ in range(n_blocks_3072)])
        self.proj2 = ProjectionBlock(mid1_dim, mid2_dim, dropout=0.1)

        blocks_low = []
        n_simple = max(0, n_blocks_2048 - n_rich_2048)
        for i in range(n_blocks_2048):
            if i < n_simple:
                blocks_low.append(ResidualBlock(mid2_dim, dropout=0.1))
            else:
                blocks_low.append(RichResidualBlock2048(dim=mid2_dim, hidden=mid2_dim * 2, dropout=0.1))
        self.low = nn.Sequential(*blocks_low)

        self.out_norm = RMSNorm(mid2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.high(x)
        x = self.proj2(x)
        x = self.low(x)
        x = self.out_norm(x)
        return F.normalize(x, p=2, dim=-1)


class SiameseHeadV3(nn.Module):
    def __init__(self, in_dim: int = 4096):
        super().__init__()
        self.head = ProjectionHeadV3(in_dim=in_dim)

    def forward(self, ref_emb: torch.Tensor, var_emb: torch.Tensor):
        z_r = self.head(ref_emb)
        z_v = self.head(var_emb)
        return z_r, z_v

"""
============================================================
제출 시 단일 시퀀스 임베딩을 위해 사용하는 버전.
state_dict 구조는 SiameseHeadV3와 동일(head.xxx)
============================================================
"""
class SiameseHeadV3Single(nn.Module):
    def __init__(
        self,
        in_dim: int = 4096,
        mid1_dim: int = 3072,
        mid2_dim: int = 2048,
        out_dim: int = 2048,
        n_blocks_3072: int = 8,
        n_blocks_2048: int = 16,
        n_rich_2048: int = 4,
    ):
        super().__init__()
        self.head = ProjectionHeadV3(
            in_dim=in_dim,
            mid1_dim=mid1_dim,
            mid2_dim=mid2_dim,
            out_dim=out_dim,
            n_blocks_3072=n_blocks_3072,
            n_blocks_2048=n_blocks_2048,
            n_rich_2048=n_rich_2048,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


'''
============================================================
초기화: v3를 v2로부터 warm-start
    - v2 head에서 proj1/proj2 + 일부 블록 weight를 v3에 복사
    - Norm / LayerScale / alpha는 v3에서 새로 학습
============================================================
'''
def init_v3_from_v2(v3: SiameseHeadV3, v2: SiameseHeadV2):
    with torch.no_grad():
        v3_head = v3.head
        v2_head = v2.head

        # proj1: linear/skip만 복사
        v3_p1 = v3_head.proj1.state_dict()
        v2_p1 = v2_head.proj1.state_dict()
        for k in v3_p1:
            if ("linear" in k or "skip" in k) and k in v2_p1 and v3_p1[k].shape == v2_p1[k].shape:
                v3_p1[k] = v2_p1[k]
        v3_head.proj1.load_state_dict(v3_p1)

        # proj2: linear/skip만 복사
        v3_p2 = v3_head.proj2.state_dict()
        v2_p2 = v2_head.proj2.state_dict()
        for k in v3_p2:
            if ("linear" in k or "skip" in k) and k in v2_p2 and v3_p2[k].shape == v2_p2[k].shape:
                v3_p2[k] = v2_p2[k]
        v3_head.proj2.load_state_dict(v3_p2)

        # high: v2.high(4개) -> v3.high 앞 4개
        n_copy_high = min(len(v2_head.high), len(v3_head.high))
        for i in range(n_copy_high):
            v3_blk_state = v3_head.high[i].state_dict()
            v2_blk_state = v2_head.high[i].state_dict()
            for k in v3_blk_state:
                if "linear" in k and k in v2_blk_state and v3_blk_state[k].shape == v2_blk_state[k].shape:
                    v3_blk_state[k] = v2_blk_state[k]
            v3_head.high[i].load_state_dict(v3_blk_state)

        # low: v2.low의 simple 블록 -> v3.low의 simple 블록 앞에서부터 (최대 8개)
        v2_simple = [b for b in v2_head.low if isinstance(b, ResidualBlockV2)]
        v3_simple = [b for b in v3_head.low if isinstance(b, ResidualBlock)]
        n_copy_low = min(len(v2_simple), len(v3_simple), 8)
        for i in range(n_copy_low):
            v3_blk_state = v3_simple[i].state_dict()
            v2_blk_state = v2_simple[i].state_dict()
            for k in v3_blk_state:
                if "linear" in k and k in v2_blk_state and v3_blk_state[k].shape == v2_blk_state[k].shape:
                    v3_blk_state[k] = v2_blk_state[k]
            v3_simple[i].load_state_dict(v3_blk_state)

        print(
            f"[Init] v3 head initialized from v2: "
            f"proj1/proj2, high {n_copy_high} blocks, low(simple) {n_copy_low} blocks."
        )


'''
============================================================
Loss (CDD + PCC)
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
        d = cosine_distance(z_r, z_v)

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
                neigh_viol = F.relu(d_sorted[:-1] - d_sorted[1:] + self.k_margin).mean()
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
Evaluation (CD / CDD / PCC + score_lb)
    - loss는 (CDD + PCC)로 보고
    - score_lb는 (CD + CDD + PCC)로 계산
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
        return {
            "loss": 0.0,
            "CD": 0.0,
            "CD_Benign": 0.0,
            "CD_Patho": 0.0,
            "CDD": 0.0,
            "PCC": 0.0,
            "score_lb": 0.0,
        }

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

    score_lb = cd + cdd + pcc

    return {
        "loss": float(total_loss / max(1, n_loss)),
        "CD": cd,
        "CD_Benign": db,
        "CD_Patho": dp,
        "CDD": cdd,
        "PCC": pcc,
        "score_lb": score_lb,
    }


'''
============================================================
Config (v3)
============================================================
'''
@dataclass
class TrainingConfig:
    emb_path: str
    out_dir: str
    batch_size: int = 4096
    epochs: int = 150
    lr: float = 2e-4


'''
============================================================
Greedy Soup (topK, alpha-mix grid)
    - epoch_infos를 score_lb 내림차순 정렬
    - 후보(epoch)들을 순회하며, 현재 soup_state와 cand_state를 alpha로 섞어 평가
    - val score_lb가 개선되면 ingredient로 채택
    - 최종 souped state 저장: head_best_souped_v3.pt
============================================================
'''
def greedy_soup_epochs(
    cfg: TrainingConfig,
    epoch_infos: List[Dict[str, Any]],
    dl_val: DataLoader,
    in_dim: int,
    device: str = "cuda",
):
    print(f"\n[v3 Epoch Soup] Starting Greedy Soup over {len(epoch_infos)} epochs...")

    eval_loss_fn = MetricsLoss(margin=2.0, k_margin=0.02)
    net = SiameseHeadV3(in_dim=in_dim).to(device)

    if len(epoch_infos) == 0:
        print("[v3 Epoch Soup] No epoch_infos provided. Skipping soup.")
        return None, None

    epoch_infos_sorted = sorted(epoch_infos, key=lambda x: x["score_lb"], reverse=True)
    total_epochs = len(epoch_infos_sorted)
    top_k = total_epochs
    top_epoch_infos = epoch_infos_sorted[:top_k]

    print(f"[v3 Epoch Soup] Using top {top_k} / {total_epochs} epochs for soup (score_lb=CD+CDD+PCC)")

    best_info = top_epoch_infos[0]
    soup_state = torch.load(best_info["path"], map_location="cpu")
    net.load_state_dict(soup_state)
    met0 = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    best_score = met0["score_lb"]
    num_ingredients = 1

    print(
        f"[v3 Epoch Soup] Initial soup from epoch {best_info['epoch']} | "
        f"score_lb={best_score:.4f} (CD={met0['CD']:.4f}, "
        f"CDD={met0['CDD']:.4f}, PCC={met0['PCC']:.4f})"
    )

    alphas = [i / 10.0 for i in range(1, 11)]  # 0.1~1.0

    for info in top_epoch_infos[1:]:
        print(f"\n[v3 Epoch Soup] Try add epoch {info['epoch']} (single score_lb={info['score_lb']:.4f})")
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
            mix_score = met["score_lb"]

            print(
                f"   alpha_mix={alpha:.2f} -> score_lb={mix_score:.4f} "
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
            print(f"   [Skip] No alpha_mix improved over current best score_lb {best_score:.4f}")

    seed_dir = Path(cfg.out_dir) / f"seed_{SEED}"
    soup_path = seed_dir / "head_best_souped_v3.pt"
    torch.save(soup_state, soup_path)

    print(f"\n[v3 Epoch Soup] Final souped model saved: {soup_path}")
    print(f"[v3 Epoch Soup] Final Val score_lb: {best_score:.4f}, ingredients={num_ingredients}")

    net.load_state_dict(soup_state)
    final_val = evaluate_epoch(net, dl_val, device, eval_loss_fn)
    with open(seed_dir / "v3_epoch_soup_val_summary.json", "w") as f:
        json.dump({"val": final_val, "score_lb": best_score}, f, indent=2)

    return soup_path, final_val


'''
============================================================
Evo2 로더 + test inference 유틸 (v3 제출용)
    - backbone: Evo2 7B 로드
    - hook: blocks.25.mlp.l3 (L26, 4096-d) 출력 캡처
    - pooling: attention_mask 기반 mean pooling
    - head: souped v3 head를 적용해 2048-d embedding 생성 후 CSV 저장
============================================================
'''
def load_evo2_model(model_path: str):
    try:
        from evo2 import Evo2, models  # noqa: F401
    except ImportError as e:
        raise RuntimeError("evo2 패키지가 필요합니다. `pip install evo2` 후 다시 시도하세요.") from e

    config_abs_path = str(CONFIG_DIR / "evo2-7b-1m.yml")
    try:
        from evo2 import models  # type: ignore
        models.CONFIG_MAP["evo2_7b"] = config_abs_path
        print(f"[Config Fix] models.CONFIG_MAP['evo2_7b'] = {config_abs_path}")
    except Exception:
        pass

    path_obj = Path(model_path)
    if path_obj.is_dir():
        pt_files = list(path_obj.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"{model_path} 내에 .pt 파일이 없습니다.")
        real_path = str(pt_files[0])
    else:
        real_path = str(path_obj)

    print(f"[Model] Loading Evo2 from: {real_path}")
    from evo2 import Evo2 as Evo2Wrapper  # type: ignore
    wrapper = Evo2Wrapper(model_name="evo2_7b", local_path=real_path)
    model = getattr(wrapper, "model", wrapper)
    tokenizer = getattr(wrapper, "tokenizer", None)
    return model, tokenizer


def encode_batch(tokenizer, seqs, max_len: int, device: str):
    # Evo2 전용 tokenizer 가정
    if hasattr(tokenizer, "tokenize"):
        padded, masks = [], []
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        for s in seqs:
            t = tokenizer.tokenize(s)[:max_len]
            pad = max_len - len(t)
            padded.append(t + [pad_id] * pad)
            masks.append([1] * len(t) + [0] * pad)

        return (
            torch.tensor(padded, device=device, dtype=torch.long),
            torch.tensor(masks, device=device, dtype=torch.long),
        )

    # HF-style tokenizer fallback
    toks = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return toks.input_ids.to(device), toks.attention_mask.to(device)


class TestDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        self.ids = df["ID"].tolist()
        self.seqs = df["seq"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.ids[idx]


def generate_submission_v3_souped(
    head_ckpt_path: Path,
    device: str,
    output_csv_path: Path,
):
    if not head_ckpt_path.exists():
        raise FileNotFoundError(f"Soup head checkpoint not found: {head_ckpt_path}")
    if not TEST_CSV_PATH.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_CSV_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Evo2 model path not found: {MODEL_PATH}")

    print(f"\n[Submission] Loading Evo2 backbone + tokenizer...")
    backbone, tokenizer = load_evo2_model(str(MODEL_PATH))

    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    backbone.to(device, dtype=dtype)
    backbone.eval()

    # Hook: blocks.25.mlp.l3 (L26, 4096-d)
    layer_name = "blocks.25.mlp.l3"
    activations: Dict[str, torch.Tensor] = {}

    def get_activation(name):
        def hook(module, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            activations[name] = val.detach()
        return hook

    hooked = False
    if hasattr(backbone, "blocks"):
        try:
            backbone.blocks[25].mlp.l3.register_forward_hook(get_activation(layer_name))
            print(f"[Hook] Registered hook on backbone.blocks[25].mlp.l3")
            hooked = True
        except Exception as e:
            print(f"[Hook] Direct hook failed: {e}")

    if not hooked:
        for n, m in backbone.named_modules():
            if n == layer_name:
                m.register_forward_hook(get_activation(layer_name))
                print(f"[Hook] Registered hook on {n}")
                hooked = True
                break

    if not hooked:
        raise RuntimeError(f"Could not find layer {layer_name} in Evo2 backbone.")

    # v3 souped head 로드
    print(f"[Head] Loading v3 souped head from: {head_ckpt_path}")
    head_net = SiameseHeadV3Single(
        in_dim=4096,
        mid1_dim=3072,
        mid2_dim=2048,
        out_dim=2048,
        n_blocks_3072=8,
        n_blocks_2048=16,
        n_rich_2048=4,
    ).to(device)

    state = torch.load(head_ckpt_path, map_location=device)
    head_net.load_state_dict(state, strict=True)
    head_net.eval()

    # Test DataLoader
    test_ds = TestDataset(TEST_CSV_PATH)
    test_dl = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[Inference] Start on {len(test_ds)} sequences for submission...")
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    with torch.no_grad():
        for seqs, ids in tqdm(test_dl, desc="Test-v3", leave=False):
            input_ids, attn_mask = encode_batch(tokenizer, seqs, max_len=1024, device=device)

            if device == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    _ = backbone(input_ids)
                    token_feats = activations[layer_name].float()  # [B, L, 4096]
                    mask_f = attn_mask.unsqueeze(-1).float()       # [B, L, 1]
            else:
                _ = backbone(input_ids)
                token_feats = activations[layer_name].float()
                mask_f = attn_mask.unsqueeze(-1).float()

            if token_feats.shape[1] != mask_f.shape[1]:
                L = min(token_feats.shape[1], mask_f.shape[1])
                token_feats = token_feats[:, :L, :]
                mask_f = mask_f[:, :L, :]

            denom = mask_f.sum(1).clamp_min(1.0)
            pooled = (token_feats * mask_f).sum(1) / denom  # [B, 4096]

            emb = head_net(pooled)  # [B, 2048] (L2-normalized)
            all_embeddings.append(emb.cpu().numpy())
            all_ids.extend(ids)

    print("[Output] Concatenating embeddings...")
    final_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 2048]

    cols = [f"emb_{i:04d}" for i in range(final_embeddings.shape[1])]
    df_out = pd.DataFrame(final_embeddings, columns=cols)
    df_out.insert(0, "ID", all_ids)

    print(f"[Output] Saving submission CSV to {output_csv_path}")
    df_out.to_csv(output_csv_path, index=False)
    print("[Output] Submission CSV saved.")


'''
============================================================
main (v3 학습 + soup + 제출)
    - Student(v3) 학습
    - Teacher(v2) 존재 시 distillation + warm-start 사용
    - epoch ckpt 저장 후, 상위 MAX_CKPT_TO_KEEP개만 디스크에 유지
    - 상위 ckpt들로 greedy soup 수행
    - souped head로 test.csv가 있으면 submission CSV 생성
============================================================
'''
def main():
    cfg = TrainingConfig(
        emb_path=str(EMB_PATH),
        out_dir=str(OUTPUT_DIR),
        batch_size=4096,
        epochs=150,
        lr=2e-4,
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

    # Student: v3
    net = SiameseHeadV3(in_dim=in_dim).to(device)

    # Teacher: v2
    teacher = SiameseHeadV2(in_dim=in_dim).to(device)
    use_teacher = TEACHER_CKPT.exists()
    if use_teacher:
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
        print("[Teacher] Loaded v2 head from", TEACHER_CKPT)
        init_v3_from_v2(net, teacher)
    else:
        print("[Teacher] WARNING: v2 teacher ckpt not found, distillation disabled.")

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
        f"\n[Train] Start (v3, seed={SEED}, epochs={cfg.epochs}, "
        f"batch={cfg.batch_size}, lr={cfg.lr})"
    )

    for epoch in range(1, cfg.epochs + 1):
        net.train()
        run_loss, n = 0.0, 0
        pbar = tqdm(dl_train, desc=f"[v3 Seed {SEED}] Ep {epoch}", leave=False)

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

            # distillation loss: epoch <= 15 까지
            if use_teacher and epoch <= 15:
                with torch.no_grad():
                    z_r_t, z_v_t = teacher(ref_t, var_t)
                l_dist = (z_r - z_r_t.detach()).pow(2).mean() + (z_v - z_v_t.detach()).pow(2).mean()
            else:
                l_dist = torch.tensor(0.0, device=device)

            # loss schedule
            if use_teacher and epoch <= 15:
                # distillation-only
                loss = l_dist
                lambda_pcc = 0.0
                lambda_dist = 1.0
            else:
                # full: CDD + PCC + CD-reg
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
        score_lb = val_met["score_lb"]

        ckpt_path = seed_dir / f"head_ep{epoch:03d}.pt"
        torch.save(net.state_dict(), ckpt_path)

        info = {
            "epoch": epoch,
            "path": str(ckpt_path),
            "score_lb": float(score_lb),
            "CD": float(val_met["CD"]),
            "CDD": float(val_met["CDD"]),
            "PCC": float(val_met["PCC"]),
        }
        epoch_infos.append(info)

        best_ckpts.append(info)
        best_ckpts.sort(key=lambda x: x["score_lb"], reverse=True)
        while len(best_ckpts) > MAX_CKPT_TO_KEEP:
            worst = best_ckpts.pop()
            if os.path.exists(worst["path"]):
                os.remove(worst["path"])

        improved = False
        if score_lb > best_single_score:
            best_single_score = score_lb
            best_single_epoch = epoch
            improved = True
            torch.save(net.state_dict(), seed_dir / "head_best_single.pt")
            with open(seed_dir / "v3_best_single_val_summary.json", "w") as f:
                json.dump(
                    {"epoch": epoch, "metrics": val_met, "score_lb": best_single_score},
                    f,
                    indent=2,
                )

        print(
            f"[v3 Seed {SEED}] Ep {epoch:03d} | "
            f"TrainLoss: {run_loss/n:.4f} | "
            f"ValLoss(CDD+PCC): {val_met['loss']:.4f} | "
            f"CD: {val_met['CD']:.4f} | "
            f"CDD: {val_met['CDD']:.4f} | "
            f"PCC: {val_met['PCC']:.4f} | "
            f"score_lb(CD+CDD+PCC): {score_lb:.4f} "
            f"(Best single: {best_single_score:.4f} @ep{best_single_epoch}{' *' if improved else ''}) | "
            f"lam_pcc={lambda_pcc:.2f}, lam_dist={lambda_dist:.2f}"
        )

    # 모든 epoch 기록 저장
    with open(seed_dir / "v3_epoch_val_metrics.json", "w") as f:
        json.dump(epoch_infos, f, indent=2)

    print(
        f"\n[Done] v3 training finished. Best single epoch = {best_single_epoch} "
        f"(score_lb={best_single_score:.4f})"
    )

    # best single ckpt로 재평가
    print("\n[Val] Evaluating v3 best single-epoch model on Val Set...")
    best_single_path = seed_dir / "head_best_single.pt"
    net.load_state_dict(torch.load(best_single_path, map_location=device))
    eval_loss_for_val = MetricsLoss(margin=2.0, k_margin=0.02)
    best_single_val = evaluate_epoch(net, dl_val, device, eval_loss_for_val)
    with open(seed_dir / "v3_best_single_val_final_summary.json", "w") as f:
        json.dump({"val": best_single_val, "score_lb": best_single_score}, f, indent=2)

    # 상위 ckpt들로 soup
    soup_path, soup_val = greedy_soup_epochs(
        cfg=cfg,
        epoch_infos=best_ckpts,
        dl_val=dl_val,
        in_dim=in_dim,
        device=device,
    )

    # 메모리 정리
    del net
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    # souped head로 제출 CSV 생성
    if soup_path is not None and TEST_CSV_PATH.exists():
        print("\n[Submission] Generating submission CSV using v3 souped head...")
        generate_submission_v3_souped(
            head_ckpt_path=soup_path,
            device=device,
            output_csv_path=SUBMISSION_CSV_PATH,
        )
    else:
        print("\n[Submission] Skipped: soup_path is None or test.csv not found.")


if __name__ == "__main__":
    main()
