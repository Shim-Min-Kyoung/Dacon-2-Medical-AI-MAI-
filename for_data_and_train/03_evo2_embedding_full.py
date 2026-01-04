import math
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm

'''
============================================================
경로 설정
    - 스크립트 위치 기준으로 프로젝트 루트를 잡고, 학습/전처리 로컬 데이터는 for_data_and_train/local_data/ 아래를 사용
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"     # 전처리/학습용 로컬 데이터 루트
WEIGHT_DIR = PROJECT_ROOT / "weights"          # 모델 체크포인트 폴더
CONFIG_DIR = PROJECT_ROOT / "configs"          # 모델 설정(YAML) 폴더

# 입력 데이터(.npz): ref/var 시퀀스(정수 ID)와 라벨/메타 포함
DATA_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_clinvar_snv_indel50.npz"
MODEL_PATH = WEIGHT_DIR / "evo2_7b"            # evo2_7b.pt가 있는 폴더

# Evo2 YAML 설정 파일(프로젝트 configs/)
CONFIG_SRC_PATH = CONFIG_DIR / "evo2-7b-1m.yml"

# 출력 임베딩(.npz): ref/var pooled embedding + 메타데이터 저장
OUT_PATH = LOCAL_DATA_DIR / "dataset/refvar_L1024_evo2L26_emb_full.npz"

print(f"[System] Input : {DATA_PATH}")
print(f"[System] Output: {OUT_PATH}")

'''
============================================================
유틸 함수
============================================================
'''
DNA_ALPHABET = np.array(list("ACGTN"), dtype="U1")

# 재현성을 위한 RNG seed 고정
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

'''
============================================================
정수 ID 배치([B, L])를 DNA 문자열 리스트(List[str])로 변환.
    Args:
        batch_ids: [B, L] 형태의 int array (0~4 => A/C/G/T/N)

    Returns:
        길이 B의 DNA 문자열 리스트
============================================================
'''
def ids_batch_to_seqs(batch_ids: np.ndarray) -> List[str]:
    chars = DNA_ALPHABET[batch_ids.astype(np.int64)]  # [B, L] -> 문자
    seqs = ["".join(row) for row in chars]
    return seqs

'''
============================================================
DNA 문자열 리스트를 토큰으로 인코딩.
    - Evo2 tokenizer에 .tokenize()가 있으면 그 방식을 사용.
    - 아니면 HuggingFace 스타일 __call__() 방식으로 인코딩.

    Returns:
        input_ids: [B, max_len]
        attn_mask: [B, max_len] (토큰 위치=1, 패딩=0)
============================================================
'''
def encode_batch_strings(tokenizer, seqs: List[str], max_len: int, device: str):
    if hasattr(tokenizer, "tokenize"):
        padded, masks = [], []
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        for s in seqs:
            t = tokenizer.tokenize(s)[:max_len]
            pad = max_len - len(t)
            padded.append(t + [pad_id] * pad)
            masks.append([1] * len(t) + [0] * pad)
        input_ids = torch.tensor(padded, device=device, dtype=torch.long)
        attn_mask = torch.tensor(masks, device=device, dtype=torch.long)
        return input_ids, attn_mask

    toks = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device=device, dtype=torch.long, non_blocking=True)
    attn_mask = toks["attention_mask"].to(device=device, dtype=torch.long, non_blocking=True)
    return input_ids, attn_mask

'''
============================================================
모델 로딩 (Evo2 + config 파일 복사)
Evo2 모델을 로드하고, 토크나이저와 backbone(실제 torch model)을 반환.

    주의할 점:
      - Evo2가 설정 파일을 상대 경로(configs/...)로 찾는 경우가 있어,
        현재 작업 디렉토리(CWD) 아래에 configs/를 만들고 YAML을 복사해둠.

    Args:
        model_path: 체크포인트 폴더 또는 .pt 파일 경로
        device: cuda device
        dtype: torch.float16 또는 torch.bfloat16

    Returns:
        backbone: torch model (eval, device/dtype 적용)
        tokenizer: Evo2 tokenizer
============================================================
'''
def load_evo2_model_for_precompute(model_path: str, device: torch.device, dtype: torch.dtype):
    try:
        from evo2 import Evo2
    except ImportError as e:
        raise RuntimeError("evo2 패키지가 없습니다. `pip install evo2`로 설치하세요.") from e

    # CWD 기준으로 configs/ 준비 (Evo2가 상대경로로 설정 파일을 읽는 경우 대응)
    cwd = Path.cwd()
    cwd_config_dir = cwd / "configs"
    cwd_config_dir.mkdir(exist_ok=True)

    cfg_dst = cwd_config_dir / "evo2-7b-1m.yml"
    if not cfg_dst.exists():
        if not CONFIG_SRC_PATH.exists():
            raise FileNotFoundError(
                f"CONFIG_SRC_PATH가 존재하지 않습니다: {CONFIG_SRC_PATH}"
            )
        shutil.copy(CONFIG_SRC_PATH, cfg_dst)
        print(f"[Config] Copied config to: {cfg_dst}")
    else:
        print(f"[Config] Using existing config at: {cfg_dst}")

    # 체크포인트 경로 해석: 폴더면 첫 *.pt 사용, 파일이면 그대로 사용
    path_obj = Path(model_path)
    if path_obj.is_dir():
        pt_files = list(path_obj.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"{model_path} 폴더 안에 .pt 파일이 없습니다.")
        real_path = str(pt_files[0])
    else:
        real_path = model_path

    print(f"[Model] Loading Evo2 from: {real_path}")
    wrapper = Evo2(model_name="evo2_7b", local_path=real_path)

    tokenizer = getattr(wrapper, "tokenizer", None)
    backbone = getattr(wrapper, "model", wrapper)

    backbone.to(device, dtype=dtype)
    backbone.eval()
    return backbone, tokenizer

'''
============================================================
메인: ID -> 문자열 -> 토큰 -> (중간 레이어) 임베딩 추출
    ref/var 시퀀스에 대해 Evo2 중간 레이어 출력에서 mean pooling 임베딩을 만들고 저장.
    Args:
        batch_size: 한 번에 처리할 샘플 수 (H100 1장 기준 32부터 권장)
        exit_layer: feature를 뽑을 transformer 블록 레이어 번호(1-indexed 느낌으로 사용)
        max_seq_len: 입력 시퀀스 최대 길이(패딩/절단 기준)
============================================================
'''
def precompute_embeddings(batch_size: int = 32, exit_layer: int = 26, max_seq_len: int = 1024):
    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다. (H100 1장 전제)")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True

    # 1. 데이터 로드
    print(f"[Data] Loading original dataset: {DATA_PATH}")
    data = np.load(DATA_PATH, allow_pickle=True)
    ref_ids = data["ref"]         # [N, L]
    var_ids = data["var"]         # [N, L]
    labels  = data["label"]
    group   = data["group"]
    var_len = data["var_len"]
    split   = data["split"]

    total_n = ref_ids.shape[0]
    print(f"[Data] Loaded Total N={total_n}")
    print(f"[Data] ref_ids shape: {ref_ids.shape}")
    print(f"[Data] var_ids shape: {var_ids.shape}")

    # 2. split 수정: test(2) → train(0), (학습용으로 test까지 합쳐 쓰는 설정일 때 사용)
    print("[Split] Original split counts:", np.bincount(split))
    split = split.copy()
    split[split == 2] = 0
    print("[Split] After merging test(2) into train(0):", np.bincount(split))

    # 3. 모델 로드
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[Model] Precision: {dtype}")
    backbone, tokenizer = load_evo2_model_for_precompute(str(MODEL_PATH), device=device, dtype=dtype)

    if tokenizer is None:
        raise RuntimeError("Evo2 wrapper에서 tokenizer를 찾지 못했습니다.")

    # 4. Hook 등록: 지정한 레이어(blocks[exit_layer-1].mlp.l3)의 출력을 저장
    layer_name = f"blocks.{exit_layer - 1}.mlp.l3"
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            val = output[0] if isinstance(output, tuple) else output
            activations[name] = val.detach()
        return hook

    if hasattr(backbone, "blocks"):
        backbone.blocks[exit_layer - 1].mlp.l3.register_forward_hook(get_activation(layer_name))
    else:
        found = False
        for name, module in backbone.named_modules():
            if name == layer_name:
                module.register_forward_hook(get_activation(layer_name))
                found = True
                break
        if not found:
            raise RuntimeError(f"{layer_name} 레이어를 모델에서 찾지 못했습니다.")

    # 5. 임베딩 계산 함수: (IDs -> seq -> tokenize -> forward -> mean pooling)
    @torch.inference_mode()
    def compute_embeddings(arr_ids: np.ndarray, label: str) -> np.ndarray:
        chunks = []
        num_samples = arr_ids.shape[0]
        num_batches = math.ceil(num_samples / batch_size)

        for start in tqdm(
            range(0, num_samples, batch_size),
            desc=f"Encoding {label}",
            total=num_batches,
        ):
            end = min(start + batch_size, num_samples)
            batch_ids = arr_ids[start:end]  # [B, L] int

            # 1) ID -> 문자열
            seqs = ids_batch_to_seqs(batch_ids)  # List[str]

            # 2) DNA 문자열 -> 토큰/마스크
            input_ids, attn_mask = encode_batch_strings(
                tokenizer, seqs, max_len=max_seq_len, device=device.type
            )

            # 3) forward 1회 + hook으로 중간 출력 회수
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                backbone(input_ids)                        
                token_feats = activations[layer_name]      

            # 4) attention mask 기반 mean pooling
            mask_f = attn_mask.unsqueeze(-1).to(token_feats.dtype)  # [B, L, 1]

            if token_feats.shape[1] != mask_f.shape[1]:
                token_feats = token_feats[:, :mask_f.shape[1], :]

            pooled = (token_feats * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)
            chunks.append(pooled.float().cpu().numpy().astype(np.float16))

        return np.concatenate(chunks, axis=0)

    # 6. ref / var 임베딩 계산
    print("[Encode] Computing ref embeddings...")
    ref_emb = compute_embeddings(ref_ids, "ref")
    print("[Encode] Computing var embeddings...")
    var_emb = compute_embeddings(var_ids, "var")

    print(f"[Result] Ref emb shape: {ref_emb.shape}")
    print(f"[Result] Var emb shape: {var_emb.shape}")

    # 7. 저장 (/workspace 50GB 볼륨에 저장됨)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        ref_emb=ref_emb,
        var_emb=var_emb,
        label=labels,
        group=group,
        var_len=var_len,
        split=split,
    )
    print(f"[Done] Saved embeddings to {OUT_PATH}")

'''
============================================================
H100 1장 기준, batch_size=32부터 시작. 
VRAM 여유되면 40~48 정도까지 올려볼 수 있음.
============================================================
'''
if __name__ == "__main__":
    precompute_embeddings(batch_size=32, exit_layer=26, max_seq_len=1024)
