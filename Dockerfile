FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder-llamacpp

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/llama.cpp.git /build

WORKDIR /build

RUN make -j

#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder
#
#RUN apt-get update && \
#    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
#    rm -rf /var/lib/apt/lists/*
#
#RUN git clone https://github.com/oobabooga/GPTQ-for-LLaMa /build
#
#WORKDIR /build
#
#RUN python3 -m venv /build/venv
#RUN . /build/venv/bin/activate && \
#    pip3 install --upgrade pip setuptools && \
#    pip3 install torch torchvision torchaudio && \
#    pip3 install -r requirements.txt
#
#
## https://developer.nvidia.com/cuda-gpus
## for a rtx 2060: ARG TORCH_CUDA_ARCH_LIST="7.5"
#ARG TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
#
#RUN . /build/venv/bin/activate && \
#    python3 setup_cuda.py bdist_wheel -d .
#
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder-webui

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/oobabooga/text-generation-webui /build

WORKDIR /build

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder-model

RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN curl -L https://huggingface.co/TheBloke/stable-vicuna-13B-GGML/resolve/main/stable-vicuna-13B.ggml.q4_3.bin -o /build/stable-vicuna-13B.ggml.q4_3.bin


FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Wing Lian <wing.lian@gmail.com>"
LABEL description="Docker image for GPTQ-for-LLaMa and Text Generation WebUI"

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl libportaudio2 libasound-dev git python3 python3-pip make g++ && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

RUN --mount=type=cache,target=/root/.cache/pip pip3 install virtualenv
RUN mkdir /app
RUN chown -R appuser:appuser /app

WORKDIR /app

COPY --from=builder-llamacpp /build/main /app

ARG WEBUI_VERSION
RUN test -n "${WEBUI_VERSION}" && git reset --hard ${WEBUI_VERSION} || echo "Using provided webui source"

RUN virtualenv /app/venv
RUN . /app/venv/bin/activate && \
    pip3 install --upgrade pip setuptools && \
    pip3 install torch torchvision torchaudio

#COPY --from=builder /build /app/repositories/GPTQ-for-LLaMa
#RUN . /app/venv/bin/activate && \
#    pip3 install /app/repositories/GPTQ-for-LLaMa/*.whl


COPY --from=builder-webui /build/extensions/api/requirements.txt /app/extensions/api/requirements.txt
COPY --from=builder-webui /build/extensions/elevenlabs_tts/requirements.txt /app/extensions/elevenlabs_tts/requirements.txt
COPY --from=builder-webui /build/extensions/google_translate/requirements.txt /app/extensions/google_translate/requirements.txt
COPY --from=builder-webui /build/extensions/silero_tts/requirements.txt /app/extensions/silero_tts/requirements.txt
COPY --from=builder-webui /build/extensions/whisper_stt/requirements.txt /app/extensions/whisper_stt/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip . /app/venv/bin/activate && cd extensions/api && pip3 install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip . /app/venv/bin/activate && cd extensions/elevenlabs_tts && pip3 install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip . /app/venv/bin/activate && cd extensions/google_translate && pip3 install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip . /app/venv/bin/activate && cd extensions/silero_tts && pip3 install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip . /app/venv/bin/activate && cd extensions/whisper_stt && pip3 install -r requirements.txt

COPY --from=builder-webui /build/requirements.txt /app/requirements.txt
RUN . /app/venv/bin/activate && \
    pip3 install -r requirements.txt

RUN cp /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so /app/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

RUN mkdir /app/cache
RUN mkdir -p /app/cache/huggingface/hub

ENV TRANSFORMERS_CACHE=/app/cache/huggingface/hub

COPY --from=builder-webui /build/. /app/
COPY entrypoint.sh /app/entrypoint.sh
RUN chown -R appuser:appuser /app && find /app -type d -exec chmod 0755 {} \;
RUN chmod +x /app/entrypoint.sh

USER appuser
COPY --from=builder-model /build/. /app/models/

ENTRYPOINT ["/app/entrypoint.sh"]
