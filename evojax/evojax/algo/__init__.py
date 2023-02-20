# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import NEAlgorithm
from .cma_wrapper import CMA
from .pgpe import PGPE
from .ars import ARS
from .simple_ga import SimpleGA
from .open_es import OpenES
from .cma_jax import CMA_ES_JAX
from .map_elites import MAPElites

Strategies = {
    "CMA": CMA,
    "PGPE": PGPE,
    "SimpleGA": SimpleGA,
    "ARS": ARS,
    "OpenES": OpenES,
    "CMA_ES_JAX": CMA_ES_JAX,
    "MAPElites": MAPElites,
}

__all__ = [
    "NEAlgorithm",
    "CMA",
    "PGPE",
    "ARS",
    "SimpleGA",
    "CMA_ES_JAX",
    "OpenES",
    "MAPElites",
    "Strategies",
]
