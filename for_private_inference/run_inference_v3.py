import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

'''
============================================================
경로/파일 구조
    - 학습 파이프라인:   MAI/for_data_and_train/...
    - Private 추론 코드: MAI/for_private_inference/run_inference_v3.py (현재 파일)
    - 대회 공식 데이터:  /data/*.csv (우선) -> 없으면 for_private_inference/data/*.csv 탐색
    - 최종 head 가중치:  for_private_inference/weights/head_best_souped_v3.pt
    - Evo2 모델/설정:    MAI/weights/evo2_7b, MAI/configs/evo2-7b-1m.yml
============================================================
'''

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

COMP_DATA_DIR = Path("/data")
LOCAL_DATA_DIR = SCRIPT_DIR / "data"

HEAD_WEIGHTS_DIR = SCRIPT_DIR / "weights"
HEAD_CKPT_PATH = HEAD_WEIGHTS_DIR / "head_best_souped_v3.pt"

EVO2_MODEL_DIR = PROJECT_ROOT / "weights" / "evo2_7b"
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "private_submission_v3.csv"

'''
============================================================
테스트 CSV 경로 자동 탐색

    우선순위:
      1) /data/*.csv  (사전순)
      2) for_private_inference/data/*.csv (사전순)

    - 각 후보 CSV에서 1행만 읽어 'ID'와 'seq' 컬럼이 모두 있는 파일을 찾음
    - 조건을 만족하는 첫 번째 파일을 사용.
============================================================
'''
def get_test_csv_path() -> Path:
    search_roots = [COMP_DATA_DIR, LOCAL_DATA_DIR]
    candidates: list[Path] = []

    for root in search_roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*.csv")))

    if not candidates:
        raise FileNotFoundError(
            "CSV 파일을 찾을 수 없습니다. /data 또는 for_private_inference/data 아래에 "
            "테스트용 CSV(ID,seq 컬럼)를 위치시켜 주세요."
        )

    valid: list[Path] = []
    for p in candidates:
        try:
            df_head = pd.read_csv(p, nrows=1)
            if {"ID", "seq"}.issubset(df_head.columns):
                valid.append(p)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "CSV 파일은 발견했지만, 'ID'와 'seq' 컬럼을 모두 포함한 파일을 찾지 못했습니다. "
            "/data 또는 for_private_inference/data 아래 CSV 형식을 확인해 주세요."
        )

    chosen = valid[0]
    print(f"[Input] Using test CSV: {chosen}")
    return chosen


'''
============================================================
v3 Head 네트워크 구성 요소
    - 학습 코드(07_train_v3.py)에서 사용한 구조를 그대로 복제
    - RMSNorm + LayerScale + alpha(잔차 게이팅) 기반의 projection/residual 블록들로 구성
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
차원 변경(d_in -> d_out)을 포함한 projection + residual 블록.
    y = skip(x) + alpha * (layerscale * f(x))
    f(x) = RMSNorm(x) -> Linear(d_in->d_out) -> GELU -> Dropout

    - main linear 가중치는 0으로 초기화(학습 초기에 잔차 경로 영향 최소화)
    - alpha는 0으로 초기화(학습 초기에 skip 경로 위주)
============================================================
'''
class ProjectionBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1, ls_init: float = 1e-2):
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.linear = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # 차원 정렬용 skip projection
        self.skip = nn.Linear(d_in, d_out, bias=False)

        # 안정적인 잔차 학습을 위한 초기화
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

'''
============================================================
동일 차원(dim) residual 블록.
    y = x + alpha * (layerscale * f(x))
    f(x) = RMSNorm(x) -> Linear(dim->dim) -> GELU -> Dropout
    - linear 가중치 0 초기화 + alpha 0 초기화로 학습 안정화
============================================================
'''
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

'''
============================================================
2048-d rich residual 블록(2-layer MLP).
    f(x) = RMSNorm(x)
           -> Linear(2048->hidden) -> GELU -> Dropout
           -> Linear(hidden->2048) -> Dropout
    y = x + alpha * (layerscale * f(x))
    - fc2(마지막 linear) 가중치를 0으로 초기화하여 학습 초기에 잔차 영향 최소화
============================================================
'''
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
2048-d rich residual 블록(2-layer MLP).
    f(x) = RMSNorm(x)
           -> Linear(2048->hidden) -> GELU -> Dropout
           -> Linear(hidden->2048) -> Dropout
    y = x + alpha * (layerscale * f(x))
    - fc2(마지막 linear) 가중치를 0으로 초기화하여 학습 초기에 잔차 영향 최소화
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
        self.high = nn.Sequential(
            *[ResidualBlock(mid1_dim, dropout=0.1) for _ in range(n_blocks_3072)]
        )
        self.proj2 = ProjectionBlock(mid1_dim, mid2_dim, dropout=0.1)

        blocks_low = []
        n_simple = max(0, n_blocks_2048 - n_rich_2048)
        for i in range(n_blocks_2048):
            if i < n_simple:
                blocks_low.append(ResidualBlock(mid2_dim, dropout=0.1))
            else:
                blocks_low.append(
                    RichResidualBlock2048(dim=mid2_dim, hidden=mid2_dim * 2, dropout=0.1)
                )
        self.low = nn.Sequential(*blocks_low)
        self.out_norm = RMSNorm(mid2_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(x)
        x = self.high(x)
        x = self.proj2(x)
        x = self.low(x)
        x = self.out_norm(x)
        return F.normalize(x, p=2, dim=-1)

'''
============================================================
제출(추론) 전용 단일 입력 head wrapper.
    - 학습 시에는 Siamese 구조로 두 입력을 비교했지만,
      제출 시에는 단일 시퀀스 임베딩만 필요하므로 single 버전만 사용.
    - state_dict는 학습 시의 head 파라미터와 동일한 키 구조(head.xxx)를 사용.
============================================================
'''
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
Evo2 로더 + 토크나이저 유틸
    - MAI/weights/evo2_7b 아래의 .pt 파일을 로드한다고 가정
    - config는 MAI/configs/evo2-7b-1m.yml 사용
    - backbone은 eval 모드로 전환
============================================================
'''
def load_evo2_model(model_root: Path, device: str, dtype: torch.dtype) -> Tuple[torch.nn.Module, Any]:
    try:
        from evo2 import Evo2 as Evo2Wrapper, models  # type: ignore
    except ImportError as e:
        raise RuntimeError("evo2 패키지가 필요합니다. `pip install evo2` 후 다시 시도하세요.") from e

    config_path = CONFIG_DIR / "evo2-7b-1m.yml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Evo2 설정 파일을 찾을 수 없습니다: {config_path} "
            f"(for_data_and_train/01_Evo2_download.py 실행 여부를 확인하세요.)"
        )

    # 일부 환경에서 CONFIG_MAP을 덮어씌우는 방식으로 config 경로를 지정
    try:
        models.CONFIG_MAP["evo2_7b"] = str(config_path)
    except Exception:
        pass

    if not model_root.exists():
        raise FileNotFoundError(
            f"Evo2 모델 디렉토리를 찾을 수 없습니다: {model_root} "
            f"(for_data_and_train/01_Evo2_download.py 실행 여부를 확인하세요.)"
        )

    if model_root.is_dir():
        pt_files = list(model_root.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"{model_root} 내에 .pt 파일이 없습니다.")
        real_path = str(pt_files[0])
    else:
        real_path = str(model_root)

    print(f"[Evo2] Loading from: {real_path}")
    wrapper = Evo2Wrapper(model_name="evo2_7b", local_path=real_path)
    backbone = getattr(wrapper, "model", wrapper)
    tokenizer = getattr(wrapper, "tokenizer", None)

    backbone.to(device, dtype=dtype)
    backbone.eval()
    return backbone, tokenizer

'''
============================================================
Evo2 tokenizer 전용 인코딩 유틸.
    - tokenizer.tokenize()가 존재하면: 수동 padding + attention mask 생성
    - 그 외에는 HuggingFace 스타일 callable tokenizer로 처리
============================================================
'''
def encode_batch(tokenizer, seqs: List[str], max_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
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

    toks = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return toks.input_ids.to(device), toks.attention_mask.to(device)

"""
===============================================================================
Test Dataset
    - 입력 CSV는 반드시 'ID', 'seq' 컬럼을 포함해야 함
===============================================================================
"""
class TestDataset(Dataset):
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        if "ID" not in df.columns or "seq" not in df.columns:
            raise ValueError(f"입력 CSV {csv_path} 는 'ID', 'seq' 컬럼을 포함해야 합니다.")
        self.ids = df["ID"].tolist()
        self.seqs = df["seq"].tolist()

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.ids[idx]



"""
===============================================================================
메인 추론 함수 (Submission 생성)
    1) 테스트 CSV 탐색/로드
    2) Evo2 backbone 로드 + 특정 레이어 activation hook 등록
    3) 토큰 레벨 feature를 마스킹 평균(pooling)하여 [B, 4096] 획득
    4) v3 head를 통해 [B, 2048] L2-normalized embedding 생성
    5) ID + embedding을 CSV로 저장
===============================================================================
"""
def run_inference_v3(
    output_csv_path: Path = DEFAULT_OUTPUT_CSV,
    batch_size: int = 32,
    max_seq_len: int = 1024,
) -> None:
    if not HEAD_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"v3 head 체크포인트를 찾을 수 없습니다: {HEAD_CKPT_PATH} "
            f"(제출 패키지에 for_private_inference/weights/head_best_souped_v3.pt 포함 여부를 확인하세요.)"
        )

    test_csv_path = get_test_csv_path()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        dtype = torch.float32

    print(f"[System] Device: {device} (dtype={dtype})")
    print(f"[Input] Test CSV : {test_csv_path}")
    print(f"[Head]  v3 ckpt  : {HEAD_CKPT_PATH}")

    # Evo2 backbone + tokenizer
    backbone, tokenizer = load_evo2_model(EVO2_MODEL_DIR, device=device, dtype=dtype)
    if tokenizer is None:
        raise RuntimeError("Evo2 tokenizer를 찾지 못했습니다.")

    # Hook: blocks.25.mlp.l3 (L26, 4096-d)
    layer_name = "blocks.25.mlp.l3"
    activations: Dict[str, torch.Tensor] = {}

    def get_activation(name: str):
        def hook(module, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            activations[name] = val.detach()

        return hook

    # 우선 blocks 접근 방식으로 hook 시도, 실패 시 named_modules 탐색
    hooked = False
    if hasattr(backbone, "blocks"):
        try:
            backbone.blocks[25].mlp.l3.register_forward_hook(get_activation(layer_name))  # type: ignore[attr-defined]
            print("[Hook] Registered hook on backbone.blocks[25].mlp.l3")
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
    print(f"[Head] Loading SiameseHeadV3Single from: {HEAD_CKPT_PATH}")
    head_net = SiameseHeadV3Single(
        in_dim=4096,
        mid1_dim=3072,
        mid2_dim=2048,
        out_dim=2048,
        n_blocks_3072=8,
        n_blocks_2048=16,
        n_rich_2048=4,
    ).to(device)

    state = torch.load(HEAD_CKPT_PATH, map_location=device)
    head_net.load_state_dict(state, strict=True)
    head_net.eval()

    # Test DataLoader
    test_ds = TestDataset(test_csv_path)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[Inference] Start on {len(test_ds)} sequences for submission...")
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    # 추론 루프
    with torch.no_grad():
        for seqs, ids in tqdm(test_dl, desc="Test-v3", leave=False):
            input_ids, attn_mask = encode_batch(tokenizer, list(seqs), max_len=max_seq_len, device=device)

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

    # CSV 저장
    print("[Output] Concatenating embeddings...")
    final_embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 2048]

    cols = [f"emb_{i:04d}" for i in range(final_embeddings.shape[1])]
    df_out = pd.DataFrame(final_embeddings, columns=cols)
    df_out.insert(0, "ID", all_ids)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Output] Saving submission CSV to {output_csv_path}")
    df_out.to_csv(output_csv_path, index=False)
    print("Submission CSV saved.")


if __name__ == "__main__":
    run_inference_v3()
