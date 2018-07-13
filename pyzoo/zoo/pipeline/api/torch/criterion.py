#
# Copyright 2018 Analytics Zoo Authors.
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
#

from bigdl.util.common import callBigDlFunc
from bigdl.util.common import JTensor
import numpy as np

from bigdl.nn.criterion import Criterion

class CPPOCriterion(Criterion):

    def __init__(self,
                 epsilon=0.3,
                 entropy_coeff=0.0,
                 kl_target=0.01,
                 init_beta=0.0,
                 bigdl_type="float"):
        super(CPPOCriterion, self).__init__(None,
                                                     bigdl_type,
                                                     epsilon,
                                                     entropy_coeff,
                                                     kl_target,
                                                     init_beta)