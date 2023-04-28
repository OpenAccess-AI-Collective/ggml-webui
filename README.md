---
title: Oobabooga Text Generation Webui
emoji: 🔥
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
license: agpl-3.0
models: 
- TheBloke/stable-vicuna-13B-GGML
---

# GGML Spaces w/ Oobabooga Text Generation Webui

This Space is based on https://github.com/Freedom-AI-Collective/ggml-webui. Brought to you by the Freedom AI Collective.

### quickstart

- Update the curl script at the bottom of the `Dockerfile` to download the appropriate GGML model and save it to the correct location
- Update the `entrypoint.sh` to use the updated GGML model
- Update the `README.md` metadata so HF can link your Space with the correct model
- commit to your changes
- configure your HuggingFace space repo `git remote add hf https://huggingface.co/spaces/<username>/<spacename>
- push your changes to HF. You may need to force push your first change: `git push -f hf main`

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
