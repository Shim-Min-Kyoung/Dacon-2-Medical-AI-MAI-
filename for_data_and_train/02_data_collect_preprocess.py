import os
import re
import glob
import gzip
import hashlib
from pathlib import Path
from typing import Optional

import requests
from tqdm.auto import tqdm

'''
============================================================
경로 및 전역 설정
   - 원본 다운로드(GRCh38 FASTA / ClinVar VCF): for_data_and_train/local_data/
   - 최종 데이터셋(.npz): for_data_and_train/local_data/dataset/
============================================================
'''
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOCAL_DATA_DIR = SCRIPT_DIR / "local_data"
LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 모든 데이터(FASTA/VCF/NPZ)는 이 루트 아래에 저장
ROOT = str(LOCAL_DATA_DIR)

# ClinVar 컷오프(YYYYMMDD, '이전/같음'만 사용)
CUTOFF = "20251109"

# 윈도우 길이 및 인델 최대 길이 제한
L = 1024
INDEL_MAX = 50

# ClinVar 필터 옵션
PASS_ONLY = False          # ClinVar에서는 True로 두면 과도하게 필터링될 수 있어 기본 False 권장
STRICT_REVSTAT = True      # CLNREVSTAT(근거 수준) 조건을 강제할지 여부

# 데이터셋 분할 비율(8:1:1)
TRAIN_PCT = 80
VAL_PCT = 10
TEST_PCT = 10
assert TRAIN_PCT + VAL_PCT + TEST_PCT == 100, "TRAIN/VAL/TEST 비율 합이 100이어야 합니다."

# 디버그 옵션
DEBUG = False
DEBUG_MAX = 200000
MAX_VARIANTS: Optional[int] = None


'''
============================================================
HTTP 세션 및 다운로드 유틸리티
    - 재시도/커넥션 풀링이 적용된 requests.Session 생성
============================================================
'''
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "MAI-Downloader/1.0"})
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        try:
            retries = Retry(
                total=5,
                connect=5,
                read=5,
                backoff_factor=0.5,
                status_forcelist=(500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "HEAD"]),
            )
        except TypeError:
            retries = Retry(
                total=5,
                connect=5,
                read=5,
                backoff_factor=0.5,
                status_forcelist=(500, 502, 503, 504),
                method_whitelist=frozenset(["GET", "HEAD"]),
            )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        pass
    return s


SESSION = make_session()

'''
============================================================
원격 파일 크기 조회
    1순위) HEAD 요청의 Content-Length
    2순위) Range 요청(bytes=0-0) 후 Content-Range 파싱
============================================================
'''
def _remote_size(url: str, timeout: int = 60, session: Optional[requests.Session] = None) -> Optional[int]:
    s = session or SESSION
    try:
        h = s.head(url, allow_redirects=True, timeout=timeout)
        h.raise_for_status()
        cl = h.headers.get("Content-Length")
        if cl:
            return int(cl)
    except Exception:
        pass
    try:
        r = s.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
        if r.status_code in (200, 206):
            cr = r.headers.get("Content-Range")  # 예) "bytes 0-0/1234567"
            if cr and "/" in cr:
                return int(cr.rsplit("/", 1)[-1])
            cl = r.headers.get("Content-Length")
            if cl:
                return int(cl)
    except Exception:
        pass
    return None

'''
============================================================
대용량 HTTP(S) 다운로드(이어받기 지원).
    동작 규칙:
      1) out_path가 이미 있고 skip_if_exists=True이면:
         - verify_size=True일 때 원격(Content-Length) 크기와 같으면 스킵(원격 크기 확인 불가여도 스킵)
      2) out_path.part가 있고 크기가 원격과 같으면 완성본으로 교체
      3) 부분파일(out_path+'.part') 존재 시 Range 이어받기 시도
         - 서버가 Range를 무시하고 200으로 응답하면 부분파일 삭제 후 처음부터 재다운로드
      4) verify_size=True인데 크기가 다르면 .part를 남겨두고 종료(재개 가능)
============================================================
'''
def stream_download(
    url: str,
    out_path: str,
    chunk: int = 1 << 20,
    timeout: int = 60,
    verify_size: bool = True,
    skip_if_exists: bool = True,
    session: Optional[requests.Session] = None,
) -> str:

    s = session or SESSION
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"

    # 0) 완성본 스킵
    if os.path.exists(out_path) and skip_if_exists:
        if not verify_size:
            return out_path
        try:
            rs = _remote_size(url, timeout=timeout, session=s)
            if rs is None or os.path.getsize(out_path) == rs:
                return out_path
        except Exception:
            return out_path

    # 1) 부분파일이 원격과 크기 동일하면 바로 교체
    if os.path.exists(tmp):
        try:
            rs = _remote_size(url, timeout=timeout, session=s)
            if rs and os.path.getsize(tmp) == rs:
                os.replace(tmp, out_path)
                return out_path
        except Exception:
            pass

    # 2) 이어받기 (부분파일 크기만큼 Range 요청)
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}

    with s.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        if pos and r.status_code == 200:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
            pos = 0

        try:
            rem = int(r.headers.get("Content-Length", "0"))
            total = pos + rem if rem else None
        except Exception:
            total = None

        pbar = tqdm(
            total=total,
            initial=pos,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(out_path),
        )
        with open(tmp, "ab" if pos else "wb") as f:
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    f.write(blk)
                    pbar.update(len(blk))
        pbar.close()

    # 3) 크기 검증(가능한 경우) 후 완성본으로 교체
    if verify_size:
        try:
            rs = _remote_size(url, timeout=timeout, session=s)
            if rs and os.path.getsize(tmp) != rs:
                print(
                    f"[warn] Size mismatch: {out_path} "
                    f"local={os.path.getsize(tmp)} remote={rs}. Keep .part for resume."
                )
                return tmp
        except Exception:
            pass

    os.replace(tmp, out_path)
    return out_path

'''
============================================================
단순 디렉토리 인덱스 HTML에서 href 링크 목록을 추출
============================================================
'''
def http_list(url: str, timeout: int = 60, session: Optional[requests.Session] = None):
    s = session or SESSION
    html = s.get(url, timeout=timeout).text
    return re.findall(r'href="([^"#?]+)"', html)

'''
============================================================
다운로드
    1) GRCh38 FASTA (NCBI)
    2) ClinVar weekly VCF (GRCh38): 컷오프 이전 최신본 자동 선택
============================================================
'''
def download_grch38_fasta(dest_dir: Optional[str] = None, session: Optional[requests.Session] = None) -> str:
    if dest_dir is None:
        dest_dir = os.path.join(ROOT, "grch38")
    s = session or SESSION
    base = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/"
    fname = "GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
    url = base + fname
    return stream_download(url, os.path.join(dest_dir, fname), session=s)

'''
============================================================
NCBI FTP에서 ClinVar weekly VCF(GRCh38) 다운로드.
    - clinvar_YYYYMMDD.vcf.gz 중 YYYYMMDD <= cutoff 조건을 만족하는 최신본을 선택한다.
    - .tbi / .md5가 존재하면 함께 다운로드한다.
============================================================
'''
def download_clinvar_weekly(
    dest_dir: Optional[str] = None,
    cutoff: str = CUTOFF,
    session: Optional[requests.Session] = None,
) -> str:
    if dest_dir is None:
        dest_dir = os.path.join(ROOT, "clinvar")
    s = session or SESSION
    base = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/"
    files = http_list(base, session=s)
    pat = re.compile(r"clinvar_(\d{8})\.vcf\.gz$")
    dated = sorted(
        [(f, m.group(1)) for f in files for m in [pat.match(f)] if m],
        key=lambda x: x[1],
    )
    pick = next((f for f, d in reversed(dated) if d <= cutoff), None)
    if not pick:
        raise RuntimeError("컷오프 이전 ClinVar 스냅샷을 못 찾았습니다.")
    vcf = stream_download(base + pick, os.path.join(dest_dir, pick), session=s)
    if pick + ".tbi" in files:
        stream_download(base + pick + ".tbi", os.path.join(dest_dir, pick + ".tbi"), session=s)
    if pick + ".md5" in files:
        stream_download(base + pick + ".md5", os.path.join(dest_dir, pick + ".md5"), session=s)
    return vcf


'''
============================================================
전처리(ClinVar + GRCh38)
   - 기준(ref) / 변이(var) 윈도우 시퀀스 생성 (길이 L)
   - 라벨 필터링(병원성/양성) 및 품질 조건 적용
   - 그룹 단위(윈도우 단위)로 안정적인 8:1:1 분할
============================================================
'''
def ensure_python_libs():
    try:
        import pysam  # noqa
        import numpy  # noqa
        from tqdm.auto import tqdm as _tqdm  # noqa
    except ImportError:
        raise ImportError("pysam, numpy, tqdm가 필요합니다. 예: `pip install pysam numpy tqdm`")

'''
============================================================
입력이 .gz면 1회만 압축 해제하고, 압축 해제된 파일 경로를 반환.
============================================================
'''
def decompress_gzip_if_needed(path: str) -> str:
    if not path.endswith(".gz"):
        return path
    out_path = path[:-3]
    if os.path.exists(out_path):
        return out_path
    print(f"[info] decompressing FASTA: {path} -> {out_path}")
    import shutil

    with gzip.open(path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out_path

'''
============================================================
ROOT/grch38 아래에서 FASTA를 찾고, .fai 인덱스가 없으면 생성한 뒤 (pysam.FastaFile, fasta_path)를 반환.
============================================================
'''
def get_fasta_handle_and_path():
    import pysam

    fa_dir = os.path.join(ROOT, "grch38")
    cand = []
    for pat in ["*.fna", "*.fa", "*.fna.gz", "*.fa.gz"]:
        cand.extend(glob.glob(os.path.join(fa_dir, pat)))
    if not cand:
        raise RuntimeError(f"FASTA not found under {fa_dir}.")
    fa_path = sorted(cand)[0]
    print("[info] FASTA candidate:", fa_path)
    fa_plain = decompress_gzip_if_needed(fa_path)
    if not os.path.exists(fa_plain + ".fai"):
        print("[info] building FASTA index (.fai)")
        pysam.faidx(fa_plain)
    fasta = pysam.FastaFile(fa_plain)
    return fasta, fa_plain

'''
============================================================
ROOT/clinvar 아래에서 날짜 <= CUTOFF인 ClinVar VCF(또는 VCF.GZ) 중 최신본을 선택.
============================================================
'''
def pick_clinvar_vcf():
    clin_dir = os.path.join(ROOT, "clinvar")
    cands = glob.glob(os.path.join(clin_dir, "clinvar_*.vcf")) + glob.glob(os.path.join(clin_dir, "clinvar_*.vcf.gz"))
    if not cands:
        raise RuntimeError(
            f"ClinVar VCF not found in {clin_dir}. 'clinvar_YYYYMMDD.vcf[.gz]' 형식인지 확인하세요."
        )
    pairs = []
    for path in cands:
        base = os.path.basename(path)
        m = re.match(r"clinvar_(\d{8})\.vcf(\.gz)?$", base)
        if not m:
            continue
        date = m.group(1)
        if CUTOFF is not None and date > CUTOFF:
            continue
        pairs.append((date, path))
    if not pairs:
        raise RuntimeError("CUTOFF 조건을 만족하는 ClinVar VCF가 없습니다.")
    pairs.sort()
    date, path = pairs[-1]
    print(f"[info] Using ClinVar VCF: {path} (date={date})")
    return path

'''
============================================================
VCF의 chrom 표기와 FASTA 컨티그 이름을 연결하는 매핑 dict 생성.
    일반적으로 발생하는 표기 차이를 흡수:
        - FASTA: 'chr1' 형태 또는 RefSeq accession('NC_000001.11') 형태
        - VCF : '1'..'22', 'X','Y', 'chr1'.., 'MT'/'chrM'/'M'

    미토콘드리아(MT):
        - 'NC_012920' 포함 또는 길이 16569bp 컨티그를 찾아 MT/chrM/M 별칭을 매핑.
============================================================
'''
def build_chrom_map(fasta):
    chrom_map = {}
    refs = list(fasta.references)

    for r in refs:
        chrom_map.setdefault(r, r)
        if r.startswith("chr"):
            chrom_map.setdefault(r[3:], r)

        m = re.match(r"NC_0*([0-9]{1,3})\.\d+", r)
        if m:
            num = int(m.group(1))
            base = None
            if 1 <= num <= 22:
                base = str(num)
            elif num == 23:
                base = "X"
            elif num == 24:
                base = "Y"
            if base is not None:
                chrom_map.setdefault(base, r)

    mt_target = None
    for r in refs:
        try:
            if "NC_012920" in r or (
                hasattr(fasta, "get_reference_length") and fasta.get_reference_length(r) == 16569
            ):
                mt_target = r
                break
        except Exception:
            pass
    if mt_target:
        for key in ("MT", "chrM", "M"):
            chrom_map.setdefault(key, mt_target)

    print("[info] chrom_map examples (first 12):")
    for i, (k, v) in enumerate(list(chrom_map.items())[:12]):
        print(f"  {k} -> {v}")
    if mt_target:
        print(f"[info] MT alias mapped to: {mt_target}")
    else:
        print("[warn] MT alias mapping not set (could not detect mitochondrial contig).")

    return chrom_map

'''
============================================================
pos(VCF 1-based)를 중심으로 길이 L의 기준(ref) 윈도우를 가져옴.
    반환: (시퀀스, window_start)  (window_start는 1-based)
============================================================
'''
def get_ref_window(fasta, chrom_fa: str, pos: int, L: int = L):
    half = L // 2
    start = max(1, pos - half)
    end = start + L - 1
    try:
        seq = fasta.fetch(chrom_fa, start - 1, end)
    except Exception:
        return None, None
    if len(seq) != L:
        return None, None
    return seq.upper(), start

# ClinVar 라벨링 규칙(이진 분류)
P = {"Pathogenic", "Likely_pathogenic"}
B = {"Benign", "Likely_benign"}
DROP = {
    "Conflicting_interpretations_of_pathogenicity",
    "Uncertain_significance",
    "not_provided",
    "no_assertion_provided",
    "no_assertion_criteria_provided",
    "risk_factor",
    "protective",
    "association",
    "drug_response",
}

'''
============================================================
(CLNSIG, CLNREVSTAT) -> 라벨 변환.
    - 1: (Likely_)Pathogenic
    - 0: (Likely_)Benign
    - -1: 그 외(모호/드롭/근거 부족)
============================================================
'''
def to_label(clnsig: str, clnrev: str) -> int:
    if not isinstance(clnsig, str):
        return -1
    toks = {t for t in re.split(r"[|/,;&\s]+", clnsig) if t}
    if toks & DROP:
        return -1
    has_p = bool(toks & P)
    has_b = bool(toks & B)
    if has_p and has_b:
        return -1
    if STRICT_REVSTAT:
        ok = isinstance(clnrev, str) and (
            "expert_panel" in clnrev
            or "practice_guideline" in clnrev
            or "criteria_provided" in clnrev
        )
        if not ok:
            return -1
    if has_p:
        return 1
    if has_b:
        return 0
    return -1

'''
============================================================
문자열이 A/C/G/T로만 구성되어 있으면 True
============================================================
'''
def is_dna4(s: str) -> bool:
    return isinstance(s, str) and all(c in "ACGT" for c in s.upper())

'''
============================================================
기준(ref) 윈도우(ref_seq, 길이 L)에 (REF->ALT) 변이를 적용해 변이(var) 윈도우를 만듦.
    처리 규칙:
        - 삽입(ALT 길이 증가): 길이가 늘어나면 L까지 자름.
        - 결실(ALT 길이 감소): 부족한 길이만큼 윈도우 끝 이후의 기준 염기를 추가로 가져와 채움.
        - 추가 fetch 실패 시 'N'으로 패딩.
============================================================
'''
def apply_variant_with_fix(
    ref_seq: str,
    fasta,
    chrom_fa: str,
    window_start: int,
    pos: int,
    REF: str,
    ALT: str,
    L: int = L,
):
    offset = pos - window_start
    if offset < 0 or offset + len(REF) > len(ref_seq):
        return None
    if ref_seq[offset : offset + len(REF)] != REF:
        return None
    var_seq = ref_seq[:offset] + ALT + ref_seq[offset + len(REF) :]
    delta = len(ALT) - len(REF)
    if delta > 0:
        var_seq = var_seq[:L]
    elif delta < 0:
        fill = -delta
        try:
            extra = fasta.fetch(chrom_fa, window_start - 1 + L, window_start - 1 + L + fill).upper()
        except Exception:
            extra = ""
        if len(extra) < fill:
            extra += "N" * (fill - len(extra))
        var_seq = (var_seq + extra)[:L]
    if len(var_seq) != L:
        return None
    return var_seq

'''
============================================================
ClinVar VCF + 기준 FASTA로부터 데이터셋을 구축
    출력 배열:
        - ref, var: (N, L) uint8 인코딩 (A,C,G,T,N -> 0..4)
        - label: (N,) int8  {0,1}
        - chrom, pos, window_start
        - group: (chrom_fa, window_start) 기준 그룹 ID(누수 방지용)
        - split: 0=train, 1=val, 2=test (그룹 단위 안정적 해시 분할)
        - var_kind: 0=SNV, 1=INS, 2=DEL
        - var_len: 인델 길이(|len(ALT)-len(REF)|), SNV는 1
============================================================
'''
def build_refvar_dataset(
    vcf_path: str,
    fasta,
    chrom_map: dict,
    max_variants: Optional[int] = None,
    debug: bool = True,
    debug_max: int = 200000,
):
    import pysam
    import numpy as np
    from tqdm.auto import tqdm

    vcf = pysam.VariantFile(vcf_path)
    print("[info] VCF header samples:", list(vcf.header.samples))

    # 필터링 단계별 카운터(디버그/진단용)
    stats = {
        "total_alt": 0,
        "fail_filter": 0,
        "fail_ref_dna4": 0,
        "fail_alt_dna4": 0,
        "fail_same_allele": 0,
        "fail_len": 0,
        "fail_label": 0,
        "fail_window": 0,
        "fail_apply": 0,
        "kept": 0,
    }
    example_keys = [
        "fail_filter",
        "fail_ref_dna4",
        "fail_alt_dna4",
        "fail_len",
        "fail_label",
        "fail_window",
        "fail_apply",
    ]
    examples = {k: [] for k in example_keys}

    # ASCII 문자 -> base id 매핑(A,C,G,T,N -> 0..4)
    BASE2ID = np.full(256, 4, dtype=np.uint8)
    for b, i in zip("ACGTN", [0, 1, 2, 3, 4]):
        BASE2ID[ord(b)] = i

    def encode_seq(seq: str):
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        return BASE2ID[arr]

    # INFO 필드를 안전하게 추출(ALT별 tuple 형태도 처리).
    def get_info_field(record, key, alt_index=None):
        val = record.info.get(key, None)
        if val is None:
            return None
        if isinstance(val, (str, bytes)):
            return val.decode() if isinstance(val, bytes) else val
        if isinstance(val, tuple):
            if len(val) == 0:
                return None
            if alt_index is not None and len(val) > alt_index:
                v = val[alt_index]
            else:
                v = val[0]
            return v.decode() if isinstance(v, bytes) else (v if isinstance(v, str) else str(v))
        return val if isinstance(val, str) else str(val)

    # 누적 리스트
    ref_list, var_list = [], []
    label_list, chrom_list, pos_list, wstart_list = [], [], [], []
    group_list, kind_list, vlen_list = [], [], []

    # (chrom_fa, window_start) 그룹 단위 분할을 위해 단위 그룹 ID 부여
    group_map = {}
    gid2key = {}
    next_gid = 0
    processed_alt = 0

    for rec in tqdm(vcf, desc="scan ClinVar VCF"):
        chrom_vcf = rec.chrom
        pos = rec.pos
        REF = rec.ref
        alts = rec.alts or ()

        chrom_fa = chrom_map.get(chrom_vcf, chrom_vcf)

        for alt_index, ALT in enumerate(alts):
            processed_alt += 1
            stats["total_alt"] += 1
            if max_variants is not None and processed_alt > max_variants:
                break
            if debug and processed_alt > debug_max:
                break

            ALT = str(ALT)
            REF_u = REF.upper()
            ALT_u = ALT.upper()

            # FILTER 조건(선택)
            if PASS_ONLY:
                if "PASS" not in rec.filter.keys():
                    stats["fail_filter"] += 1
                    if len(examples["fail_filter"]) < 5:
                        examples["fail_filter"].append(
                            {"chrom_vcf": chrom_vcf, "chrom_fa": chrom_fa, "pos": pos, "filter": list(rec.filter.keys())}
                        )
                    continue
            
            # REF/ALT가 A/C/G/T로만 구성된 경우만 허용
            if not is_dna4(REF_u):
                stats["fail_ref_dna4"] += 1
                if len(examples["fail_ref_dna4"]) < 3:
                    examples["fail_ref_dna4"].append({"REF": REF, "ALT": ALT})
                continue
            if not is_dna4(ALT_u):
                stats["fail_alt_dna4"] += 1
                if len(examples["fail_alt_dna4"]) < 3:
                    examples["fail_alt_dna4"].append({"REF": REF, "ALT": ALT})
                continue
            
            # REF == ALT는 제외
            if REF_u == ALT_u:
                stats["fail_same_allele"] += 1
                continue

            # 변이 종류 분류(SNV vs 인델)
            if len(REF_u) == 1 and len(ALT_u) == 1:
                kind = 0
                vlen = 1
            else:
                delta_len = abs(len(ALT_u) - len(REF_u))
                if delta_len == 0 or delta_len > INDEL_MAX:
                    stats["fail_len"] += 1
                    if len(examples["fail_len"]) < 3:
                        examples["fail_len"].append({"REF": REF, "ALT": ALT, "delta_len": delta_len})
                    continue
                kind = 1 if len(ALT_u) > len(REF_u) else 2
                vlen = delta_len

            # ClinVar 라벨 생성
            clnsig = get_info_field(rec, "CLNSIG", alt_index=alt_index)
            clnrev = get_info_field(rec, "CLNREVSTAT", alt_index=alt_index)
            label = to_label(clnsig, clnrev)
            if label not in (0, 1):
                stats["fail_label"] += 1
                if len(examples["fail_label"]) < 5:
                    examples["fail_label"].append(
                        {
                            "chrom_vcf": chrom_vcf,
                            "chrom_fa": chrom_fa,
                            "pos": pos,
                            "clnsig": clnsig,
                            "clnrev": clnrev,
                        }
                    )
                continue

            # 기준(ref) 윈도우 가져오기
            ref_seq, wstart = get_ref_window(fasta, chrom_fa, pos, L=L)
            if ref_seq is None:
                stats["fail_window"] += 1
                if len(examples["fail_window"]) < 3:
                    examples["fail_window"].append(
                        {
                            "chrom_vcf": chrom_vcf,
                            "chrom_fa": chrom_fa,
                            "pos": pos,
                            "REF": REF_u,
                            "ALT": ALT_u,
                        }
                    )
                continue

            # 변이(var) 윈도우 생성(REF->ALT 적용)
            var_seq = apply_variant_with_fix(ref_seq, fasta, chrom_fa, wstart, pos, REF_u, ALT_u, L=L)
            if var_seq is None:
                stats["fail_apply"] += 1
                if len(examples["fail_apply"]) < 3:
                    examples["fail_apply"].append(
                        {
                            "chrom_vcf": chrom_vcf,
                            "chrom_fa": chrom_fa,
                            "pos": pos,
                            "REF": REF_u,
                            "ALT": ALT_u,
                        }
                    )
                continue

            # 시퀀스 인코딩(A,C,G,T,N -> 0..4)
            ref_arr = encode_seq(ref_seq)
            var_arr = encode_seq(var_seq)
            if ref_arr.shape[0] != L or var_arr.shape[0] != L:
                continue

            # (chrom_fa, window_start) 분할 누수 방지를 위해 단위로 그룹 ID 부여
            key = (chrom_fa, wstart)
            gid = group_map.get(key)
            if gid is None:
                gid = next_gid
                group_map[key] = gid
                gid2key[gid] = key
                next_gid += 1

            ref_list.append(ref_arr)
            var_list.append(var_arr)
            label_list.append(label)
            chrom_list.append(chrom_vcf)
            pos_list.append(pos)
            wstart_list.append(wstart)
            group_list.append(gid)
            kind_list.append(kind)
            vlen_list.append(vlen)
            stats["kept"] += 1

        # 디버그 조기 종료
        if max_variants is not None and processed_alt > max_variants:
            break
        if debug and processed_alt > debug_max:
            print(f"[debug] early stop after {processed_alt} ALT records for debugging.")
            break

    # 디버그 출력
    print("\n[debug] filter statistics:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v:,}")
    print("\n[debug] example failures:")
    for k in example_keys:
        exs = examples[k]
        if not exs:
            continue
        print(f"  {k}: {len(exs)} examples")
        for e in exs:
            print("    ", e)

    if stats["kept"] == 0:
        raise RuntimeError("조건을 만족하는 변이가 하나도 없습니다. 위 디버그 통계를 참고해서 필터 조건을 조정하세요.")

    # 배열로 스택/변환
    ref_arr = np.stack(ref_list, axis=0).astype("uint8")
    var_arr = np.stack(var_list, axis=0).astype("uint8")
    labels = np.array(label_list, dtype="int8")
    chroms = np.array(chrom_list, dtype=object)
    poses = np.array(pos_list, dtype="int64")
    wstarts = np.array(wstart_list, dtype="int64")
    groups = np.array(group_list, dtype="int64")
    kinds = np.array(kind_list, dtype="int8")
    vlens = np.array(vlen_list, dtype="int16")

    # 그룹 단위 안정적(재현 가능한) 해시 버킷 분할
    def stable_bucket(chrom_fa, wstart):
        key = f"{chrom_fa}:{int(wstart)}".encode()
        h = hashlib.md5(key).hexdigest()
        return int(h[:8], 16) % 100

    g2bucket = {}
    for gid, (cfa, ws) in gid2key.items():
        g2bucket[int(gid)] = stable_bucket(cfa, ws)

    split = []
    for g in groups:
        b = g2bucket[int(g)]
        if b < TRAIN_PCT:
            split.append(0)
        elif b < TRAIN_PCT + VAL_PCT:
            split.append(1)
        else:
            split.append(2)
    split = np.array(split, dtype="int8")

    print(
        "[info] final dataset size: {} (train {:,} / val {:,} / test {:,})".format(
            len(labels),
            int((split == 0).sum()),
            int((split == 1).sum()),
            int((split == 2).sum()),
        )
    )
    print(
        "[info] kind counts (0:SNV,1:INS,2:DEL):",
        {k: int((kinds == k).sum()) for k in [0, 1, 2]},
    )

    return {
        "ref": ref_arr,
        "var": var_arr,
        "label": labels,
        "chrom": chroms,
        "pos": poses,
        "window_start": wstarts,
        "group": groups,
        "split": split,
        "var_kind": kinds,
        "var_len": vlens,
    }


'''
============================================================
실행 엔트리 포인트
============================================================
'''
def main():
    ensure_python_libs()

    print("== 1) GRCh38 FASTA 다운로드 ==")
    grch = download_grch38_fasta()

    print("== 2) ClinVar (GRCh38 weekly, 컷오프 이전 최신본) 다운로드 ==")
    clinvar_vcf = download_clinvar_weekly()

    from time import sleep

    print("\n== 3) ClinVar + GRCh38 전처리 (ref/var 윈도우 + 8:1:1 스플릿) ==")
    fasta, fa_plain = get_fasta_handle_and_path()
    chrom_map = build_chrom_map(fasta)
    vcf_path = pick_clinvar_vcf()

    dataset = build_refvar_dataset(
        vcf_path,
        fasta,
        chrom_map,
        max_variants=MAX_VARIANTS,
        debug=DEBUG,
        debug_max=DEBUG_MAX,
    )

    import numpy as np

    out_dir = os.path.join(ROOT, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"refvar_L{L}_clinvar_snv_indel{INDEL_MAX}.npz")
    np.savez_compressed(out_path, **dataset)

    print("\n==== DONE ====")
    print("GRCh38 FASTA:", grch)
    print("ClinVar VCF :", clinvar_vcf)
    print("Saved dataset:", out_path)
    for k in ["ref", "var", "label", "group", "split", "var_kind", "var_len"]:
        a = dataset[k]
        print(f"  {k:10s}: {getattr(a, 'shape', None)} {getattr(a, 'dtype', None)}")


if __name__ == "__main__":
    main()
