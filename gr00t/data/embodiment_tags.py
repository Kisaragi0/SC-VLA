# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """
    XARM6 = "xarm6"
    XARM7 = "xarm7"
    PANDA = "panda"
    WIDOWXAI = "widowxai" 

    ARX5 = "arx5"


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.XARM6.value: 3,
    EmbodimentTag.PANDA.value: 2,
    EmbodimentTag.XARM7.value: 1,
    # EmbodimentTag.WIDOWXAI.value: 0,
    EmbodimentTag.ARX5.value: 0,
}
