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

from zoo.common.nncontext import *
from zoo.util.engine import prepare_env
from pyspark.files import SparkFiles

prepare_env()
if not SparkFiles._is_running_on_worker:
    check_version()
JavaCreator.add_creator_class("com.intel.analytics.zoo.pipeline.nnframes.python.PythonNNFrames")
JavaCreator.add_creator_class("com.intel.analytics.zoo.feature.python.PythonImageFeature")
JavaCreator.add_creator_class("com.intel.analytics.zoo.pipeline.api.keras.python.PythonAutoGrad")
JavaCreator.add_creator_class("com.intel.analytics.zoo.models.python.PythonZooModel")
JavaCreator.add_creator_class("com.intel.analytics.zoo.pipeline.api.keras.python.PythonZooKeras2")
