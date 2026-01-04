'''
============================================================
RunPod 전용: Evo2 7B 모델 및 설정 파일 다운로드 스크립트
 - Hugging Face에서 모델 가중치 다운로드
 - Evo2용 YAML 설정 파일 생성
 - 다운로드 무결성(용량) 검증
# ============================================================
'''
import os
from pathlib import Path

# 1. 필수 라이브러리 설치 (huggingface_hub)  - 미설치 시 자동 설치
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    os.system("pip install -U huggingface_hub")
    from huggingface_hub import hf_hub_download

# 2. 경로 설정 (프로젝트 루트 기준, 모델/설정은 weights/ 및 configs/ 사용)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "weights" / "evo2_7b"
CONFIG_DIR = PROJECT_ROOT / "configs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

print(f" 저장 경로 설정 완료: {PROJECT_ROOT}")

'''
============================================================
Evo2 YAML 설정 파일 생성
============================================================
'''
yaml_content = """model_name: shc-evo2-7b-8k-2T-v2

vocab_size: 512
hidden_size: 4096
num_filters: 4096
hcl_layer_idxs: [2,6,9,13,16,20,23,27,30]
hcm_layer_idxs: [1,5,8,12,15,19,22,26,29]
hcs_layer_idxs: [0,4,7,11,14,18,21,25,28]
attn_layer_idxs: [3,10,17,24,31]

hcm_filter_length: 128
hcl_filter_groups: 4096
hcm_filter_groups: 256
hcs_filter_groups: 256
hcs_filter_length: 7
num_layers: 32

short_filter_length: 3 
num_attention_heads: 32
short_filter_bias: false
mlp_init_method: torch.nn.init.zeros_
mlp_output_init_method: torch.nn.init.zeros_
eps: 0.000001
state_size: 16
rotary_emb_base: 10000
rotary_emb_scaling_factor: 128
use_interpolated_rotary_pos_emb: True
make_vocab_size_divisible_by: 8
inner_size_multiple_of: 16
inner_mlp_size: 11264
log_intermediate_values: False
proj_groups: 1
hyena_filter_groups: 1
column_split_hyena: False
column_split: True
interleave: True
evo2_style_activations: True
model_parallel_size: 1
pipe_parallel_size: 1
tie_embeddings: True
mha_out_proj_bias: True
hyena_out_proj_bias: True
hyena_flip_x1x2: False
qkv_proj_bias: False
use_fp8_input_projections: True
max_seqlen: 1048576
max_batch_size: 1
final_norm: True 
use_flash_attn: True
use_flash_rmsnorm: False
use_flash_depthwise: False
use_flashfft: False
use_laughing_hyena: False
inference_mode: True
tokenizer_type: CharLevelTokenizer 
prefill_style: fft
mlp_activation: gelu
print_activations: False
"""

yaml_path = CONFIG_DIR / "evo2-7b-1m.yml"
yaml_path.write_text(yaml_content)
print(f" YAML 파일 생성 완료: {yaml_path}")


'''
============================================================
Hugging Face에서 Evo2 7B 모델 다운로드
============================================================
'''
repo_id = "arcinstitute/evo2_7b"
files_to_download = ["evo2_7b.pt", "config.json"]

print(f"\n⬇️ 모델 다운로드 시작 ({repo_id})... (RunPod 속도로 빠르게 진행됩니다)")

for filename in files_to_download:
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False  # 심볼릭 링크 끄기 (실제 파일 저장)
        )
        print(f"  -> 다운로드 성공: {filename}")
    except Exception as e:
        print(f"   다운로드 실패 ({filename}): {e}")

'''
============================================================
모델 파일 용량 검증 (1KB 오류 방지)
============================================================
'''
pt_file = MODEL_DIR / "evo2_7b.pt"

if pt_file.exists():
    size_gb = pt_file.stat().st_size / (1024**3)
    print(f"\n [검증 결과] {pt_file.name}")
    print(f"   크기: {size_gb:.2f} GB")
    
    if size_gb > 10:
        print("\n  성공! 13GB 정상 파일이 확인되었습니다.")
        print(f"   모델 경로: {MODEL_DIR}")
        print(f"   설정 경로: {yaml_path}")
    else:
        print("\n 실패! 파일 크기가 너무 작습니다. (1KB 문제 재발)")
else:
    print("\n 오류: 모델 파일이 존재하지 않습니다.")
