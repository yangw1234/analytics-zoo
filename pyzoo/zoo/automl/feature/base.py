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

from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
        Abstract Base class for basic transformers.
    """

    @abstractmethod
    def transform(self, inputs):
        """
        transform data with the input
        :param inputs: input to be transformed
        :return: transformed inputs
        """
        raise NotImplementedError()


class InversibleTransformer(BaseTransformer):
    """
        Abstract class for transformers that can be inverse transformed.
    """

    @abstractmethod
    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        raise NotImplementedError()


class BaseEstimator(InversibleTransformer):
    """
        Abstract class for basic estimators.
    """
    @abstractmethod
    def fit(self, inputs):
        raise NotImplementedError()

    def fit_transform(self, inputs):
        """
        fit and transform inputs
        :param inputs: input to be fitted
        :return: transformed inputs
        """
        return self.fit(inputs).transform(inputs)

    @abstractmethod
    def save(self, file_path):
        """
        save the states of estimator
        :param file_path
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def restore(self, file_path):
        """
        restore the states from file
        :param file_path
        :return:
        """
        raise NotImplementedError()
