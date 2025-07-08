# Stage 1: Builder
# CUDA 개발 환경을 사용하여 의존성을 빌드합니다.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV TZ=Etc/UTC

# 시스템 업데이트 및 빌드에 필요한 모든 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    autoconf \
    automake \
    pkg-config \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 가상 환경 생성 및 활성화
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# llama-cpp-python을 CUDA 지원으로 빌드하기 위한 환경 변수 설정
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs"

# Link the CUDA stub library to a place where the linker can find it.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1

# requirements.txt를 복사하고 모든 의존성을 설치합니다.
# llama-cpp-python은 소스에서 컴파일됩니다.
COPY requirements.txt .
RUN pip install --no-cache-dir --no-binary=llama-cpp-python -r requirements.txt


# Stage 2: Final Image
# CUDA 런타임 환경을 기반으로 최종 이미지를 생성합니다.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV TZ=Etc/UTC

# 런타임에 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Builder 스테이지에서 생성된 가상 환경을 복사
COPY --from=builder /opt/venv /opt/venv

# 작업 디렉토리 설정 및 PATH 환경 변수 설정
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

# 애플리케이션 소스 코드 복사
COPY . .

# Python 로그가 버퍼링 없이 즉시 출력되도록 설정
ENV PYTHONUNBUFFERED=1

# FastAPI 애플리케이션이 사용할 포트(8000)를 외부에 노출
EXPOSE 8000

# 컨테이너 실행 시 uvicorn 서버를 시작하는 기본 명령어
CMD ["python", "-m", "uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]