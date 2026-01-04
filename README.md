## 1. 프로젝트 개요

- Evo2 7B 유전체 언어모델(gLM)과 ClinVar/GRCh38 데이터를 활용해 변이(ref/var) 쌍의 거리 기반 점수를 학습하는 프로젝트입니다.
- 최종 제출 모델은 Evo2 7B 기반 임베딩 + v3 Siamese head(`head_best_souped_v3.pt`)를 사용합니다.
- 본 레포는 **데이터 수집/전처리/학습 전체 파이프라인**과 **Private score 복원용 추론 코드**를 포함하는 것을 목표로 합니다.

---

## 2. 환경 요구사항

- GPU
  - NVIDIA H100 80GB VRAM 이상 **필수** (임베딩/대규모 배치 학습 시, 패키지 설치)
- NVIDIA Driver + CUDA 런타임 12.8 계열
- **CUDA Toolkit (devel)**:
  - nvcc 사용 가능
  - CUDA 헤더(`crt/host_defines.h` 등) 포함 (`CUDA_HOME=/usr/local/cuda` 등으로 설정)
- Python 3.12
- PyTorch: `torch==2.8.0` (개발환경에서는 `2.8.0+cu128` 사용, 따라서 `2.8.0+cu128`을 권장)


본 코드들에서 사용하는 Evo2 / flash_attn / TransformerEngine 의 CUDA 확장 모듈 빌드 및 로드는  
**CUDA Toolkit이 포함된 GPU 서버 환경**에서만 정상 동작합니다.  
WSL/CPU-only 환경에서는 `flash_attn`, `transformer_engine` 설치가 실패할 수 있으며,  
실제 학습/Private Score 복원은 RunPod(H100)와 같은 GPU 환경을 전제로 합니다.


- 디스크
  - 최소 100GB 이상 **권장**
- 네트워크
  - **인터넷 권장**  
    - NCBI(ClinVar, GRCh38), Hugging Face(Evo2)에서 데이터 및 모델 다운로드 진행 시 인터넷 환경 필수
- 소프트웨어
  - OS: Linux (Ubuntu 계열 기준으로 검증)
  - Python: 3.12.x
  - 주요 라이브러리 버전: `requirements.txt` 참고  
    - `evo2==0.4.0`, `transformers==4.57.3`, `torch==2.8.0+cu128`, `numpy==2.1.2`, `pandas==2.3.3` 등

> 환경 세팅은 `for_data_and_train/00_env_setup.py` 실행으로 구성할 수 있습니다.

---

## 3. 폴더 구조

- `for_data_and_train/`  
  - 데이터 수집 → 전처리 → gLM 임베딩 생성 → v0~v3 head 학습까지 **전체 파이프라인 코드**가 순서대로 들어 있습니다.
  - 사용 순서:
    1. `00_env_setup.py`  : 시스템/파이썬 패키지 설치
    2. `01_Evo2_download.py` : Evo2 7B 모델 및 config 다운로드
    3. `02_data_collect_preprocess.py` : ClinVar/GRCh38 다운로드 및 학습용 npz 생성
    4. `03_evo2_embedding_full.py` : Evo2로 ref/var 임베딩 사전 계산
    5. `04_train_v0.py` ~ `07_train_v3.py` : v0~v3 head 학습

- `for_data_and_train/local_data/`  
  - 학습/전처리용 **로컬 데이터** 저장 경로 (외부 데이터, npz, 임베딩 등).  
  - `for_data_and_train/*.py` 가 자동으로 사용하는 경로입니다.

- `weights/` (선택적으로 사용할 수 있는 공통 모델 디렉토리)  
  - Evo2 7B base gLM 등 모델 파일을 둘 수 있는 경로입니다.
  - `for_data_and_train/01_Evo2_download.py` 실행 시 `weights/evo2_7b/evo2_7b.pt` 같은 구조를 사용할 수 있습니다.

- `for_private_inference/`  
  - **Private score 복원용 추론 코드 및 리소스**를 두는 폴더입니다.
  - `for_private_inference/data/` : 로컬에서 테스트/Private 데이터를 둘 수 있는 위치.
  - `for_private_inference/weights/` : 최종 제출에 사용하는 head 가중치
    - `head_best_souped_v3.pt` (최종 v3 모델; Private 복원에 실제 사용)

- `requirements.txt`  
  - 개발/실행 환경에 사용된 주요 라이브러리 및 버전 정보.

---

## 4. 외부 데이터 증빙

- `00_env_setup`, `01_Evo2_download.py`, `02_data_collect_preprocess.py` 파일들에서 코드 단위로 url과 라이브러리, 모델 및 데이터의 사용방식이 포함되어 있습니다. 아래는 스냅샷으로 해당 내용들을 정리해둔 것입니다.

### 4.1 ClinVar (GRCh38 weekly VCF)

- 출처: NCBI ClinVar
- 라이선스 : NCBI에 속해있는 공개데이터로, 자유롭게 사용 가능합니다. 아래 링크는 ClinVar가 공개데이터라는 것에 대한 레퍼런스입니다.
  - 레퍼런스 : `https://www.ncbi.nlm.nih.gov/clinvar/intro/`
- 사용 데이터: GRCh38 기반 weekly VCF (컷오프 날짜 `CUTOFF <= 2025-11-09`)
- 공식 URL:
  - `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/`
- 사용 코드:
  - `for_data_and_train/02_data_collect_preprocess.py`
    - `download_clinvar_weekly()` : 컷오프 이전 최신 weekly VCF 선택 및 다운로드
    - `build_refvar_dataset()` : ClinVar 변이 기반 ref/var 윈도우 및 라벨(npz) 생성
- 사용 목적:
  - ClinVar 병원성(Pathogenic) / 양성(Benign) 변이로부터 학습용 라벨 데이터셋 구축

### 4.2 GRCh38 reference genome (FASTA)

- 출처: NCBI
- 라이선스 : NCBI genebank 내 인간 게놈데이터로, 공개데이터입니다. 아래 링크는 GRCh38가 공개데이터라는 것에 대한 레퍼런스입니다.
  - 레퍼런스 : `https://www.ncbi.nlm.nih.gov/genbank/about`
- 사용 데이터: `GCF_000001405.40_GRCh38.p14_genomic.fna.gz`
- 공식 URL:
  - `https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/`
- 사용 코드:
  - `for_data_and_train/02_data_collect_preprocess.py`
    - `download_grch38_fasta()` : FASTA 다운로드
    - `get_fasta_handle_and_path()` : 컨티그 접근 및 ref 서열 추출
- 사용 목적:
  - ClinVar 변이 주변 reference 서열(ref window) 추출 후, ref/var 시퀀스 윈도우 구성

### 4.3 Evo2 7B gLM
- 라이선스: Apache-2.0 (상업적 이용 허용)
  - Hugging Face 모델 카드 상 `license: apache-2.0` 명시

- 출처: Hugging Face  
  - 리포지토리: `arcinstitute/evo2_7b`
- 사용 코드:
  - `for_data_and_train/01_Evo2_download.py`
    - Hugging Face Hub에서 `evo2_7b.pt`, `config.json` 다운로드
  - `for_data_and_train/03_evo2_embedding_full.py`
    - `load_evo2_model_for_precompute()` : Evo2 로딩
    - ref/var 시퀀스를 Evo2 토크나이저로 인코딩 → 특정 레이어(예: 26층)의 토큰 임베딩 mean pooling → ref/var 임베딩 npz 생성
- 사용 목적:
  - 유전체 언어모델(gLM)로 ref/var 서열의 고차원 표현을 얻고, 해당 임베딩 위에 Siamese head를 학습

- 출시: 2025년 2월
- 마지막 업데이트: 2025년 10월
- 본 레포에서는 2025-11-09 이전 공개된 체크포인트(evo2_7b, Apache-2.0)를 사용하며,
  이후 버전/업데이트는 사용하지 않았습니다.

> 위 항목들 외에 추가적인 외부 데이터는 사용하지 않았습니다.

---

## 5. gLM 및 Head 모델 증빙

- Base gLM
  - 모델: Evo2 7B (`arcinstitute/evo2_7b`)
  - 라이브러리: `evo2==0.4.0`
  - 다운로드/로드: `for_data_and_train/01_Evo2_download.py`, `for_data_and_train/03_evo2_embedding_full.py`

- Head 모델 (Siamese projection head)
  - v0: `for_data_and_train/04_train_v0.py`
    - 최종 souped 모델: `ckpt_v0/seed_2025/head_best_souped_v0.pt`
  - v1: `for_data_and_train/05_train_v1.py`
    - 최종 souped 모델: `ckpt_v1_seed2025_100ep/seed_2025/head_best_souped_v1.pt`
  - v2: `for_data_and_train/06_train_v2.py`
    - 최종 souped 모델: `ckpt_v2_alpha_rich/seed_2025/head_best_souped_v2.pt`
  - v3 (최종 제출용):
    - 학습 코드: `for_data_and_train/07_train_v3.py`
    - 최종 souped 모델: `ckpt_v3/seed_2025/head_best_souped_v3.pt`
    - 제출/Private 복원용 복사본: `for_private_inference/weights/head_best_souped_v3.pt`

---

## 6. 재현 및 학습 절차

1. **환경 설정**
   - Python 환경 생성 후:
     ```bash
     pip install -r requirements.txt
     ```
   - 또는 `for_data_and_train/00_env_setup.py`**(권장)** 실행으로 시스템/파이썬 패키지 자동 설치.
    - PyTorch(`torch`)는 GPU/드라이버 환경에 따라 별도 설치가 필요할 수 있으며, 개발 시에는 `torch==2.8.0+cu128` (H100, CUDA 12.8 기준) 환경에서 검증했습니다.
	- 반드시 torch 2.8.0 이상이 준비돼있어야하며 가급적  `torch==2.8.0+cu128`을 권장합니다.
	- 환경에 따라 'flash_attn'과 'transformer_engine' 패키지가 설치되지 않을 수 있습니다. 반드시 필요한 패키지이므로 cuda toolkit(dev), H100 이상 gpu를 구비하여 'flash_attn'과 'transformer_engine' 패키지를 설치하여야 합니다. 

2. **Evo2 gLM 및 설정 다운로드**
   - 실행:
     ```bash
     python 'for_data_and_train/01_Evo2_download.py'
     ```
   - 결과:
     - `weights/evo2_7b/evo2_7b.pt`
     - `configs/evo2-7b-1m.yml`

3. **ClinVar / GRCh38 데이터 수집 및 전처리 (학습/전처리 환경 준비)**
   - 실행:
     ```bash
     python 'for_data_and_train/02_data_collect_preprocess.py'
     ```
   - 결과:
     - `local_data/grch38/...`
     - `local_data/clinvar/...`
     - `local_data/dataset/refvar_L1024_clinvar_snv_indel50.npz` 등

4. **Evo2 임베딩 사전 계산 (학습/전처리 환경 준비)**
   - 실행:
     ```bash
     python 'for_data_and_train/03_evo2_embedding_full.py'
     ```
   - 결과:
     - `local_data/dataset/refvar_L1024_evo2L26_emb_full.npz`

5. **Siamese head 학습 (증류학습을 진행하므로 완벽 재현을 위해서는 모두 실행하는 것을 권장합니다.)**
   - v0 학습:
     ```bash
     python 'for_data_and_train/04_train_v0.py'
     ```
   - v1 학습:
     ```bash
     python 'for_data_and_train/05_train_v1.py'
     ```
   - v2 학습:
     ```bash
     python 'for_data_and_train/06_train_v2.py'
     ```
   - v3 학습:
     ```bash
     python 'for_data_and_train/07_train_v3.py'
     ```
   - 위 단계에서 최종 souped head는 각 `ckpt_*` 디렉토리에 저장되며,  
     제출용으로는 `head_best_souped_v3.pt`만 `weights/` 아래에 두고 사용합니다.

---

## 7. Private Score 복원 절차

사전 조건(한 번만 실행 필요):
- `for_data_and_train/00_env_setup.py`**(권장)** 또는 `pip install -r requirements.txt` 로 GPU/패키지 환경 준비
- `for_data_and_train/01_Evo2_download.py` 로 `weights/evo2_7b/`, `configs/evo2-7b-1m.yml` 준비

1. **공식 데이터 배치**
   - 대회에서 제공된 테스트/Private 데이터를 **`/data` 경로에 마운트**하거나,
   - 로컬 디버그 시에는 `for_private_inference/data/` 아래에 CSV를 둘 수 있습니다.
   - 파일명은 자유지만, CSV에는 반드시 `ID`, `seq` 컬럼이 포함되어야 합니다.
   - **현재 `data` 폴더에 `test.csv` 파일을 포함시켜 두었습니다.**

2. **최종 head 가중치 확인**
   - `for_private_inference/weights/head_best_souped_v3.pt` 존재.

3. **Private score 복원용 추론 코드 실행**
   - `for_private_inference/` 폴더 내 추론 스크립트 실행  
     (예: `run_inference_v3.py`):
     ```bash
     cd for_private_inference
     python run_inference_v3.py
     ```
   - 동작 개요:
     - `/data/*.csv` 또는 `for_private_inference/data/*.csv` 중에서 `ID, seq` 컬럼을 가진 첫 번째 CSV를 자동으로 선택하여 테스트/Private 데이터를 읽고,
     - `for_private_inference/weights/head_best_souped_v3.pt` 및 Evo2 7B를 로드,
     - 제출 형식에 맞는 예측 결과(`for_private_inference/private_submission_v3.csv`)를 생성합니다.

4. **Private Score 복원**
   - 위에서 생성된 제출 파일을 대회 플랫폼에 업로드하면,  
     본 제출 당시의 Private score를 재현할 수 있습니다.

---

## 8. 예상 소요 시간 (H100 80GB 1장 기준)

- 데이터 수집 및 전처리 (`02_data_collect_preprocess.py`)
  - ClinVar + GRCh38 다운로드 및 npz 생성: **약 10분**

- Evo2 임베딩 사전 계산 (`03_evo2_embedding_full.py`)
  - `refvar_L1024_clinvar_snv_indel50.npz` → Evo2 L26 임베딩: **약 32시간**

- Siamese head 전체 학습 파이프라인 (`04_train_v0.py` ~ `07_train_v3.py`)
  - v0~v3 전체를 순차 실행할 경우: **총 10–15시간 내외**
  - 단일 최종 모델(v3)만 다시 학습 시: v3 스크립트 기준 **7-8 시간 수준**

- private 복원 추론 시 **약 15~20분**
---

## 9. 개발 환경 스냅샷 (Runpod)

아래 환경에서 개발/실행을 검증했습니다(Runpod GPU 인스턴스 기준). 앞서 기재한 주요 라이브러리 외 버전 파악이 필요하다면 아래 스냅샷에서 확인 부탁드립니다.

```text
============================================================
[System] OS / Python 정보
============================================================
Python executable : /usr/local/bin/python
Python version    : 3.12.3
OS               : Linux 6.8.0-56-generic
OS version       : #58-Ubuntu SMP PREEMPT_DYNAMIC Fri Feb 14 15:33:28 UTC 2025
Machine          : x86_64
Processor        : x86_64

============================================================
[System] GPU / CUDA 정보 (torch 기준)
============================================================
torch version    : 2.8.0+cu128
CUDA available   : True
CUDA device      : NVIDIA H100 80GB HBM3
CUDA capability  : 9.0
Total memory     : 79.19 GB
torch.version.cuda : 12.8
cuDNN version      : 91002

============================================================
[Python] 주요 패키지 버전
============================================================
evo2              : 0.4.0
transformers      : 4.57.3
accelerate        : 1.12.0
huggingface_hub   : 0.36.0
flash_attn        : 2.8.3
transformer_engine: 2.10.0
torch             : 2.8.0+cu128
numpy             : 2.1.2
pandas            : 2.3.3
pysam             : 0.23.3
datasets          : 4.4.1
hf_transfer       : 0.1.9
tqdm              : 4.67.1
scipy             : 1.16.3
requests          : 2.32.5

============================================================
[Project] import된 라이브러리 버전 목록
============================================================

[Python] 설치된 모든 패키지 (총 201개)
============================================================
accelerate                     1.12.0
aiohappyeyeballs               2.6.1
aiohttp                        3.13.2
aiosignal                      1.4.0
annotated-types                0.7.0
anyio                          4.11.0
argon2-cffi                    25.1.0
argon2-cffi-bindings           25.1.0
arrow                          1.3.0
asttokens                      3.0.0
async-lru                      2.0.5
attrs                          25.4.0
autocommand                    2.2.2
babel                          2.17.0
backports.tarfile              1.2.0
beautifulsoup4                 4.14.2
biopython                      1.86
bleach                         6.2.0
blinker                        1.7.0
certifi                        2025.10.5
cffi                           2.0.0
charset-normalizer             3.4.3
click                          8.3.1
comm                           0.2.3
cryptography                   41.0.7
datasets                       4.4.1
dbus-python                    1.3.2
debugpy                        1.8.17
decorator                      5.2.1
defusedxml                     0.7.1
dill                           0.4.0
distlib                        0.4.0
distro                         1.9.0
einops                         0.8.1
evo2                           0.4.0
executing                      2.2.1
fastjsonschema                 2.21.2
filelock                       3.20.0
flash-attn                     2.8.3
fqdn                           1.5.1
frozenlist                     1.8.0
fsspec                         2024.6.1
h11                            0.16.0
hf-transfer                    0.1.9
hf-xet                         1.2.0
httpcore                       1.0.9
httplib2                       0.20.4
httpx                          0.28.1
huggingface-hub                0.36.0
idna                           3.10
importlib-metadata             8.7.0
inflect                        7.3.1
ipykernel                      6.30.1
ipython                        9.6.0
ipython-pygments-lexers        1.1.1
ipywidgets                     8.1.7
isoduration                    20.11.0
jaraco.collections             5.1.0
jaraco.context                 5.3.0
jaraco.functools               4.0.1
jaraco.text                    3.12.1
jedi                           0.19.2
jinja2                         3.1.6
json5                          0.12.1
jsonpointer                    3.0.0
jsonschema                     4.25.1
jsonschema-specifications      2025.9.1
jupyter-archive                3.4.0
jupyter-client                 8.6.3
jupyter-core                   5.8.1
jupyter-events                 0.12.0
jupyter-lsp                    2.3.0
jupyter-server                 2.17.0
jupyter-server-terminals       0.5.3
jupyterlab                     4.4.9
jupyterlab-pygments            0.3.0
jupyterlab-server              2.27.3
jupyterlab-widgets             3.0.15
lark                           1.3.0
launchpadlib                   1.11.0
lazr.restfulclient             0.14.6
lazr.uri                       1.0.6
markdown-it-py                 4.0.0
markupsafe                     3.0.3
matplotlib-inline              0.1.7
mdurl                          0.1.2
mistune                        3.1.4
ml-dtypes                      0.5.4
more-itertools                 10.3.0
mpmath                         1.3.0
multidict                      6.7.0
multiprocess                   0.70.18
nbclient                       0.10.2
nbconvert                      7.16.6
nbformat                       5.10.4
nest-asyncio                   1.6.0
networkx                       3.3
notebook                       7.4.2
notebook-shim                  0.2.4
numpy                          2.1.2
nvidia-cublas-cu12             12.8.4.1
nvidia-cuda-cupti-cu12         12.8.90
nvidia-cuda-nvrtc-cu12         12.8.93
nvidia-cuda-runtime-cu12       12.8.90
nvidia-cudnn-cu12              9.10.2.21
nvidia-cufft-cu12              11.3.3.83
nvidia-cufile-cu12             1.13.1.3
nvidia-curand-cu12             10.3.9.90
nvidia-cusolver-cu12           11.7.3.90
nvidia-cusparse-cu12           12.5.8.93
nvidia-cusparselt-cu12         0.7.1
nvidia-nccl-cu12               2.27.3
nvidia-nvjitlink-cu12          12.8.93
nvidia-nvtx-cu12               12.8.90
oauthlib                       3.2.2
onnx                           1.20.0
onnx-ir                        0.1.12
onnxscript                     0.5.6
packaging                      25.0
pandas                         2.3.3
pandocfilters                  1.5.1
parso                          0.8.5
pexpect                        4.9.0
pillow                         11.0.0
pip                            25.3
platformdirs                   4.5.0
prometheus-client              0.23.1
prompt-toolkit                 3.0.52
propcache                      0.4.1
protobuf                       6.33.2
psutil                         7.1.0
ptyprocess                     0.7.0
pure-eval                      0.2.3
pyarrow                        22.0.0
pycparser                      2.23
pydantic                       2.12.5
pydantic-core                  2.41.5
pygments                       2.19.2
PyGObject                      3.48.2
PyJWT                          2.7.0
pyparsing                      3.1.1
pysam                          0.23.3
python-apt                     2.7.7+ubuntu5
python-dateutil                2.9.0.post0
python-json-logger             4.0.0
pytz                           2025.2
pyyaml                         6.0.3
pyzmq                          27.1.0
referencing                    0.36.2
regex                          2025.11.3
requests                       2.32.5
rfc3339-validator              0.1.4
rfc3986-validator              0.1.1
rfc3987-syntax                 1.1.0
rich                           14.2.0
rpds-py                        0.27.1
safetensors                    0.7.0
scipy                          1.16.3
Send2Trash                     1.8.3
setuptools                     80.9.0
shellingham                    1.5.4
six                            1.16.0
sniffio                        1.3.1
soupsieve                      2.8
stack-data                     0.6.3
sympy                          1.13.3
terminado                      0.18.1
tinycss2                       1.4.0
tokenizers                     0.22.1
tomli                          2.0.1
torch                          2.8.0+cu128
torchaudio                     2.8.0+cu128
torchvision                    0.23.0+cu128
tornado                        6.5.2
tqdm                           4.67.1
traitlets                      5.14.3
transformer-engine             2.10.0
transformer-engine-cu12        2.10.0
transformer-engine-torch       2.10.0
transformers                   4.57.3
triton                         3.4.0
typeguard                      4.3.0
typer-slim                     0.20.0
types-python-dateutil          2.9.0.20251008
typing-extensions              4.15.0
typing-inspection              0.4.2
tzdata                         2025.2
uri-template                   1.3.0
urllib3                        2.5.0
virtualenv                     20.34.0
vtx                            1.0.7
wadllib                        1.3.6
wcwidth                        0.2.14
webcolors                      24.11.1
webencodings                   0.5.1
websocket-client               1.9.0
wheel                          0.45.1
widgetsnbextension             4.0.14
xxhash                         3.6.0
yarl                           1.22.0
zipp                           3.23.0
```

위 스냅샷은 개발 시점의 Runpod 환경이며, 다른 환경에서는 `requirements.txt` 기준으로 호환 가능한 버전을 설치한 뒤 실행할 수 있습니다.
