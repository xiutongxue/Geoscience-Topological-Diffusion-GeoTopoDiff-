# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

import os

import numpy as np
import yaml
from PIL import Image


def txtread(path):
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def yamlread(path):
    return yaml.safe_load(txtread(path=path))


def imwrite(img, path):
    # 确保 img 是 numpy 数组
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # 【核心修复】把多余的单通道维度挤掉
    if img.ndim == 3:
        if img.shape[-1] == 1:  # 如果是 (H, W, 1)
            img = np.squeeze(img, axis=-1)
        elif img.shape[0] == 1:  # 如果是 (1, H, W)
            img = np.squeeze(img, axis=0)

    # 如果挤掉之后还是三维 (C, H, W) 且 C=3，记得转成 (H, W, C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # 确保数据类型是 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    Image.fromarray(img).save(path)
