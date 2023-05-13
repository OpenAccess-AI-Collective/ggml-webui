FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder-llamacpp

RUN apt-get update && \
    apt-get install --no-install-recommends -y git vim build-essential python3 python3-pip python3-dev python3-venv libblas-dev liblapack-dev libopenblas-dev cmake && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install scikit-build

RUN git clone --depth 1 --branch v0.1.49 https://github.com/abetlen/llama-cpp-python.git /build
RUN git clone https://github.com/ggerganov/llama.cpp.git /build/vendor/llama.cpp

WORKDIR /build

RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 python3 setup.py bdist_wheel
# dist/llama_cpp_python-0.1.49-cp310-cp310-linux_x86_64.whl


FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL maintainer="Wing Lian <wing.lian@gmail.com>"

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential curl libportaudio2 libasound-dev git python3 python3-pip make g++ libblas-dev liblapack-dev libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser -m -d /home/appuser

RUN --mount=type=cache,target=/root/.cache/pip pip3 install virtualenv
RUN mkdir /app
RUN mkdir -p /opt/venv
RUN chown -R appuser:appuser /app

WORKDIR /app

RUN virtualenv /opt/venv
RUN . /opt/venv/bin/activate && \
    pip3 install --upgrade pip setuptools && \
    pip3 install torch torchvision torchaudio

COPY requirements.txt /app/requirements.txt
RUN . /opt/venv/bin/activate && \
    pip3 install -r requirements.txt

RUN cp /opt/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so /opt/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cpu.so

COPY --from=builder-llamacpp /build/dist/llama_cpp_python-0.1.49-cp310-cp310-linux_x86_64.whl /app
RUN . /opt/venv/bin/activate && \
    pip3 uninstall llama_cpp_python && \
    pip3 install /app/llama_cpp_python-0.1.49-cp310-cp310-linux_x86_64.whl && \
    rm /app/llama_cpp_python-0.1.49-cp310-cp310-linux_x86_64.whl

COPY . /app/

RUN mkdir -p /opt/cache/huggingface/hub
RUN chown -R appuser:appuser /app && find /app -type d -exec chmod 0755 {} \;
RUN chown -R appuser:appuser /home/appuser
RUN chmod +x /app/entrypoint.sh && \
    chmod +x /app/app.py

ENV TRANSFORMERS_CACHE=/opt/cache/huggingface/hub

USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]