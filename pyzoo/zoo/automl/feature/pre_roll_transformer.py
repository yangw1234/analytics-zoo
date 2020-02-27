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
import os

from zoo.automl.feature.base import BaseEstimator, InversibleTransformer, BaseTransformer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib
import numpy as np
from zoo.automl.feature.feature_utils import check_input_array
from sklearn.exceptions import NotFittedError


class SklearnScaler(BaseEstimator):
    """
    Transform features by scaling each feature to a given range.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.scaler = MinMaxScaler(feature_range, copy)
        self.scaler_filename = "scaler.save"
        self.fitted = False

    def fit(self, inputs):
        """
        compute the minimum and maximum to be used for later scaling
        :param inputs: array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        :return: self. Transformer instance.
        """
        check_input_array(inputs)
        self.scaler.fit(inputs)
        self.fitted = True
        return self

    def transform(self, inputs, transform_cols='all'):
        """
        scale features of inputs
        :param inputs: array-like of shape (n_samples, n_features).
            Input data that will be transformed.
        :param transform_cols: columns to be transformed. Not used in MinMaxScale.
        :return: array-like of shape (n_samples, n_features).
            Transformed data
        """
        msg = ("This %s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this method.")
        if not self.fitted:
            raise NotFittedError(msg % type(self).__name__)
        self.scaler.transform(inputs)

    def inverse_transform(self, target, transform_cols=None):
        """
        Undo the scaling of X according to feature_range.
        :param target: array-like of shape (n_samples, target_columns)
            Input data that will be transformed. It cannot be sparse.
        :param transform_cols: number of columns need to be transformed. If not None, we only apply inverse transform
            on the first transform_cols of target array.
        :return: array-like of shape (n_samples, target_columns)
            Transformed data.
        """
        assert self.scaler.scale_ is not None and len(self.scaler.scale_) > 0
        scaler_dim = len(self.scaler.scale_)
        check_input_array(target)
        if target.ndim > 2:
            raise ValueError("Expected input array has less than 3 dimensions. Got %s" % target.ndim)
        if transform_cols is not None:
            transform_cols = min(transform_cols, target.shape[1])
        if target.ndim == 2 \
                and (transform_cols is None and target.shape[1] > scaler_dim) \
                or (transform_cols > scaler_dim):
            raise ValueError("The input array has larger columns than the fitted scaler.")
        dummy_data = np.zeros(shape=(len(target), scaler_dim))
        valid_col_num = target.shape[1]
        dummy_data[, :valid_col_num] = target
        unscaled_data = self.scaler.inverse_transform(dummy_data)
        unscaled_data = unscaled_data[:valid_col_num]
        return unscaled_data

    def save(self, file_path):
        """
        save scaler into file
        :param file_path
        :return:
        """
        joblib.dump(self.scaler, os.path.join(file_path, self.scaler_filename))

    def restore(self, file_path):
        """
        restore scaler from file
        :param file_path
        :return:
        """
        self.scaler = joblib.load(os.path.join(file_path, self.scaler_filename))


class MinMaxScale(SklearnScaler):
    def __init__(self):
        self.scaler = MinMaxScaler()
        super(MinMaxScale, self).__init__()


class StandardScale(SklearnScaler):
    def __init__(self):
        self.scaler = StandardScaler
        super(StandardScale, self).__init__()


class FeatureGenerator(BaseTransformer):
    """
    generate features for input data frame
    """
    def __init__(self, generate_feature_names=None):
        """
        Constructor.
        :param generate_feature_names: a subset of
            {"month", "weekday", "day", "hour", "is_weekend", "IsAwake", "IsBusyHours"}
        """
        self.generated_feature_names = generate_feature_names

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        fit data with the input
        :param inputs: input data frame
        :param transform_cols: columns to be added into output
        :param is_train: indicate whether in training mode
        :return: numpy array. shape is (len(inputs), len(transform_cols) + len(features))
        """
        pass

    @staticmethod
    def get_allowed_features():
        return {"month", "weekday", "day", "hour", "is_weekend", "IsAwake", "IsBusyHours"}


class DeTrending(PreRollTransformer):

    def __init__(self):
        self.trend = None

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace detrending transform_cols of input data frame.
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        if is_train:
            self.trend = "detrending output"
        detrended_values = inputs[transform_cols].values - self.trend
        output_df = pd.DataFrame(data=detrended_values, columns=transform_cols)
        return output_df

    def inverse_transform(self, target):
        """
        add trend for target data
        :param target: target to be inverse transformed
        :return:
        """
        pass


class Deseasonalizing(PreRollTransformer):

    def __init__(self):
        self.seasonality = None

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace deseasonalizing transform_cols of input data frame.
        :param inputs: input input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        if is_train:
            self.seasonality = "deseasonalizing output"
        deseasonalized_values = inputs[transform_cols].values - self.seasonality
        output_df = pd.DataFrame(data=deseasonalized_values, columns=transform_cols)
        return output_df

    def inverse_transform(self, target):
        """
        add seasonality for target data
        :param target: target to be inverse transformed
        :return:
        """
        pass


class LogTransformer(PreRollTransformer):

    def transform(self, inputs, transform_cols=None, is_train=False):
        """
        inplace log transforming transform_cols of input data frame.
        :param inputs: input data frame
        :param transform_cols: columns to be transformed.
        :param is_train: indicate whether in training mode
        :return:
        """
        pass

    def inverse_transform(self, target):
        """
        inverse transform target value into origin magnitude
        :param target: target to be inverse transformed
        :return:
        """
        pass


PRE_ROLL_ORDER = {"min_max_norm": 1,
                  "standard_norm": 2,
                  "log_transfrom": 3,
                  "detrending": 4,
                  "deseasonalizing": 5,
                  "feature_generator": 6}
PRE_ROLL_TRANSFORMER_NAMES = set(PRE_ROLL_ORDER.keys())

PRE_ROLL_NAME2TRANSFORMER = {"min_max_norm": MinMaxNormalizer,
                             "standard_norm": StandardNormalizer,
                             "log_transfrom": LogTransformer,
                             "detrending": DeTrending,
                             "deseasonalizing": Deseasonalizing,
                             "feature_generator": FeatureGenerator}

PRE_ROLL_SAVE = (MinMaxNormalizer, StandardNormalizer)
PRE_ROLL_INVERSE = (MinMaxNormalizer, StandardNormalizer,
                    LogTransformer, DeTrending, Deseasonalizing)

TRANSFORM_TARGET_COL = (DeTrending, Deseasonalizing)
