import os
import sys
import subprocess
import time
import importlib.metadata

'''
============================================================
공통 유틸: 패키지 설치
    1) 특정 버전 설치 시도
    2) 실패 시 최신(호환) 버전 자동 설치 (Fallback)
============================================================
'''
def install_package(package_name, version=None, options=""):
    if version:
        cmd_strict = f'{sys.executable} -m pip install {options} "{package_name}=={version}"'
        print(f"설치 시도: {package_name} == {version}")
        try:
            subprocess.check_call(cmd_strict, shell=True)
            print(f"   성공: {package_name} == {version}")
            return
        except subprocess.CalledProcessError:
            print(f"   실패: {version} 버전을 찾을 수 없습니다. (환경 차이)")
            print(f"   대체 시도: {package_name} (최신 호환 버전) 설치 중...")
    else:
        print(f"설치 시도: {package_name} (버전 미지정 -> 최신)")

    cmd_latest = f'{sys.executable} -m pip install {options} "{package_name}"'
    try:
        subprocess.check_call(cmd_latest, shell=True)
        print(f"   성공: {package_name} (Latest/Compatible)")
    except subprocess.CalledProcessError as e:
        print(f"   오류: {package_name} 설치 실패. {e}")

'''
============================================================
torch 전용 설치 로직
    torch 설치 우선순위:
        1) torch==2.8.0+cu128
        2) torch==2.8.0
        3) torch (버전 미지정)
============================================================
'''
def install_torch():
    candidates = ["2.8.0+cu128", "2.8.0"]

    for v in candidates:
        cmd = f'{sys.executable} -m pip install "torch=={v}"'
        print(f" torch 설치 시도: {v}")
        try:
            subprocess.check_call(cmd, shell=True)
            print(f"    성공: torch=={v}")
            return
        except subprocess.CalledProcessError:
            print(f"    실패: torch=={v} (다음 후보로 진행)")

    # 2.8.0 계열이 둘 다 실패한 경우, 최신 버전 설치 시도
    print("   대체 시도: torch (버전 미지정, 최신 호환 버전)")
    try:
        subprocess.check_call(f'{sys.executable} -m pip install "torch"', shell=True)
        print("   성공: torch (Latest/Compatible)")
    except subprocess.CalledProcessError as e:
        print(f"   오류: torch 설치 실패. {e}")

'''
============================================================
전체 환경 세팅
============================================================
'''
def setup_environment():
    print("[Setup] 최종 라이브러리 설치 (matplotlib, gdown, peft 제외)")
    start_time = time.time()

    # 1. 임시 빌드 디렉토리 설정 (빌드 실패 방지)
    cwd = os.getcwd()
    tmp_dir = os.path.join(cwd, "pip_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir

    # 2. 시스템 필수 도구 (Linux 기준)
    print("\n[1/5] 시스템 빌드 도구 설치")
    sys_cmds = [
        "apt-get update -qq",
        "apt-get install -y -qq cmake git ninja-build"
    ]
    for cmd in sys_cmds:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            pass 

    # 3. pip 및 빌드 관련 패키지 업데이트
    print("\n[2/5] pip 업데이트")
    subprocess.run(f"{sys.executable} -m pip install -q --upgrade pip packaging wheel setuptools", shell=True)

    # 4. 라이브러리 설치
    print("\n[3/5] 핵심 및 유틸리티 라이브러리 설치")

    # (1) 모델 / 학습 관련 로그 기반 하드코딩
    install_torch()
    install_package("flash_attn", "2.8.3", options="--no-build-isolation --no-cache-dir")
    install_package("transformer_engine", "2.9.0")
    install_package("evo2", "0.4.0")
    install_package("transformers", "4.57.3")
    install_package("accelerate", "1.12.0")
    install_package("huggingface_hub", "0.36.0")

    # (2) 데이터 처리 및 유틸리티
    install_package("numpy", "2.1.2")
    install_package("pandas", "2.3.3")
    install_package("pysam", "0.23.3")
    install_package("requests", "2.32.5")
    install_package("hf_transfer")
    install_package("einops")
    install_package("datasets")
    
    # (3) 필수 런타임 유틸리티 
    install_package("tqdm")   # 코드 실행 필수
    install_package("scipy")  # embedding.py에 import 존재하여 유지 (필요 없으면 삭제 가능)

    # 5. Transformer Engine PyTorch 재설치 (호환성 보정)
    print("\n[4/5] Transformer Engine 호환성 보정")
    try:
        subprocess.run(f"{sys.executable} -m pip uninstall -y transformer_engine", shell=True)
        subprocess.run(f'{sys.executable} -m pip install --no-build-isolation "transformer_engine[pytorch]"', shell=True)
    except Exception:
        pass

    # 6. 완료 메시지
    print("\n[5/5] 설치 완료")
    mins = (time.time() - start_time) / 60
    print(f" 모든 설정이 완료되었습니다! (소요 시간: {mins:.1f}분)")

if __name__ == "__main__":
    setup_environment()

'''
============================================================
설치된 버전 확인
============================================================
'''
def check_versions():
    target_packages = [
        "flash_attn", "transformer_engine", "evo2", "transformers",
        "accelerate", "huggingface_hub", "numpy", "pandas",
        "pysam", "requests", "hf_transfer", "einops",
        "datasets", "tqdm", "scipy"
    ]

    print("\n" + "="*50)
    print(" [Version Check] 설치된 라이브러리 버전 확인")
    print("="*50)

    for pkg in target_packages:
        try:
            ver = importlib.metadata.version(pkg)
            print(f"✅ {pkg:<25} : {ver}")
        except importlib.metadata.PackageNotFoundError:
            print(f"❌ {pkg:<25} : 설치되지 않음")
    
    print("="*50 + "\n")

'''
============================================================
실행 엔트리
============================================================
'''
if __name__ == "__main__":
    check_versions()