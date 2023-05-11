FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder-llamacpp

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3 python3-pip python3-dev python3-venv libblas-dev liblapack-dev libopenblas-dev cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install scikit-build

RUN git clone https://github.com/abetlen/llama-cpp-python.git /build
RUN git clone https://github.com/ggerganov/llama.cpp.git /build/vendor/llama.cpp

WORKDIR /build

RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 python3 setup.py bdist_wheel
# dist/llama_cpp_python-0.1.48-cp310-cp310-linux_x86_64.whl

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

ARG OOBA_GIT_SHA=7169316

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/oobabooga/text-generation-webui /build
RUN cd /build && \
    git checkout $OOBA_GIT_SHA

WORKDIR /build

FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Wing Lian <wing.lian@gmail.com>"
LABEL description="Docker image for GPTQ-for-LLaMa and Text Generation WebUI"

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl libportaudio2 libasound-dev git python3 python3-pip make g++ libblas-dev liblapack-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

RUN --mount=type=cache,target=/root/.cache/pip pip3 install virtualenv
RUN mkdir /app
RUN chown -R appuser:appuser /app

WORKDIR /app

#ARG WEBUI_VERSION
#RUN test -n "${WEBUI_VERSION}" && git reset --hard ${WEBUI_VERSION} || echo "Using provided webui source"

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

COPY --from=builder-llamacpp /build/dist/llama_cpp_python-0.1.48-cp310-cp310-linux_x86_64.whl /app
RUN pip3 uninstall llama_cpp_python && \
    pip3 install /app/llama_cpp_python-0.1.48-cp310-cp310-linux_x86_64.whl && \
    rm /app/llama_cpp_python-0.1.48-cp310-cp310-linux_x86_64.whl

RUN mkdir /app/cache
RUN mkdir -p /app/cache/huggingface/hub

ENV TRANSFORMERS_CACHE=/app/cache/huggingface/hub

RUN pip install pyyaml
COPY --from=builder-webui /build/. /app/
COPY entrypoint.sh /app/entrypoint.sh
COPY models.yml /app/models.yml
COPY server_spaces.py /app/server_spaces.py
RUN chown -R appuser:appuser /app && find /app -type d -exec chmod 0755 {} \;
RUN chmod +x /app/entrypoint.sh

USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
