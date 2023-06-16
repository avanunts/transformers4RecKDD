#!/bin/bash
git clone https://github.com/rapidsai/rapidsai-csp-utils.git
cd rapidsai-csp-utils
git checkout patch-22.12
cd ..
python rapidsai-csp-utils/colab/env-check.py
python rapidsai-csp-utils/colab/pip-install.py

pip install -U git+https://github.com/NVIDIA-Merlin/models.git@release-23.04
pip install -U git+https://github.com/NVIDIA-Merlin/nvtabular.git@release-23.04
pip install -U git+https://github.com/NVIDIA-Merlin/core.git@release-23.04
pip install -U git+https://github.com/NVIDIA-Merlin/system.git@release-23.04
pip install -U git+https://github.com/NVIDIA-Merlin/dataloader.git@release-23.04
pip install -U git+https://github.com/NVIDIA-Merlin/Transformers4Rec.git@release-23.04
pip install -U xgboost lightfm implicit
pip install --upgrade accelerate
pip install transformers==4.28.0
pip install torchmetrics