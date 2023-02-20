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

from .base import PolicyNetwork
from .mlp import MLPPolicy
from .mlp_bis import MLPPolicy_b
from .mlp_pi import PermutationInvariantPolicy
from .convnet import ConvNetPolicy
from .metaRNN import MetaRnnPolicy
from .metaRNN_bis import MetaRnnPolicy_b
from .metaRNN_ter import MetaRnnPolicy_t
from .symla1layer import SymLA_Policy
from .metaRNN_2layers import MetaRnnPolicy_b2


__all__ = ['PolicyNetwork', 'MLPPolicy','MLPPolicy_b', 'PermutationInvariantPolicy',
           'ConvNetPolicy', 'Seq2seqPolicy','MetaRnnPolicy','MetaRnnPolicy_b','MetaRnnPolicy_t','SymLA_Policy']
