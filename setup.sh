#!/bin/bash

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install flash-attn==2.5.8 --no-build-isolation