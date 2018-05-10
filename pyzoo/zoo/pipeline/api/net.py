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

import sys

from bigdl.nn.layer import Model
from bigdl.util.common import callBigDlFunc, to_list

if sys.version >= '3':
    long = int
    unicode = str


class Net(Model):

    def __init__(self, input, output, jvalue=None, **kwargs):
        super(Model, self).__init__(jvalue,
                                    to_list(input),
                                    to_list(output),
                                    **kwargs)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Net([], [], jvalue=jvalue)
        model.value = jvalue
        return model

    @staticmethod
    def loadModel(modelPath, weightPath=None, bigdl_type="float"):
        """
        Load a pre-trained Bigdl model.

        :param path: The path containing the pre-trained model.
        :return: A pre-trained model.
        """
        jmodel = callBigDlFunc(bigdl_type, "netLoadBigDLModule", modelPath, weightPath)
        return Net.from_jvalue(jmodel)

    def new_graph(self, outputs):
        value = callBigDlFunc(self.bigdl_type, "newGraph", self.value, outputs)
        return self.from_jvalue(value)

    def freeze_up_to(self, names):
        callBigDlFunc(self.bigdl_type, "freezeUpTo", self.value, names)

    def unfreeze(self, names=None):
        callBigDlFunc(self.bigdl_type, "unFreeze", self.value, names)
