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

import ray
from ray import tune
from copy import deepcopy

from zoo.automl.search.base import *
from zoo.automl.common.util import *
from zoo.automl.common.metrics import Evaluator
from zoo.automl.common.parameters import DEFAULT_LOGGER_NAME
from ray.tune import Trainable, Stopper
from zoo.automl.logger import TensorboardXLogger
from zoo.automl.model import ModelBuilder
from zoo.orca.automl import hp
from zoo.zouwu.feature.identity_transformer import IdentityTransformer
from zoo.zouwu.preprocessing.impute import LastFillImpute, FillZeroImpute
import pandas as pd


class RayTuneSearchEngine(SearchEngine):
    """
    Tune driver
    """

    def __init__(self,
                 logs_dir="",
                 resources_per_trial=None,
                 name="",
                 remote_dir=None,
                 ):
        """
        Constructor
        :param logs_dir: local dir to save training results
        :param resources_per_trial: resources for each trial
        :param name: searcher name
        :param remote_dir: checkpoint will be uploaded to remote_dir in hdfs
        """
        self.train_func = None
        self.resources_per_trial = resources_per_trial
        self.trials = None
        self.remote_dir = remote_dir
        self.name = name
        self.logs_dir = os.path.abspath(os.path.expanduser(logs_dir))

    def compile(self,
                data,
                model_create_func,
                recipe,
                search_space=None,
                search_alg=None,
                search_alg_params=None,
                scheduler=None,
                scheduler_params=None,
                feature_transformers=None,
                mc=False,
                metric="mse"):
        """
        Do necessary preparations for the engine
        :param data: data dictionary
               Pandas Dataframe API keys:
                   "df": dataframe for training
                   "val_df": (optional) dataframe for validation
                   "feature_cols": (optional) column name for extra features
                   "target_col": (optional) column name for target
               Numpy ndarray API keys:
                   "x": ndarray for training input
                   "y": ndarray for training output
                   "x_val": (optional) ndarray for validation input
                   "y_val": (optional) ndarray for validation output
               Note: For Pandas Dataframe API keys, if "feature_cols" or "target_col" is missing,
                     then feature_transformers is required.
        :param model_create_func: model creation function
        :param recipe: search recipe
        :param search_space: search_space, required if recipe is not provided
        :param search_alg: str, all supported searcher provided by ray tune
               (i.e."variant_generator", "random", "ax", "dragonfly", "skopt",
               "hyperopt", "bayesopt", "bohb", "nevergrad", "optuna", "zoopt" and
               "sigopt")
        :param search_alg_params: extra parameters for searcher algorithm
        :param scheduler: str, all supported scheduler provided by ray tune
        :param scheduler_params: parameters for scheduler
        :param feature_transformers: feature transformer instance
        :param mc: if calculate uncertainty
        :param metric: metric name
        """

        # data mode detection
        assert isinstance(data, dict), 'ERROR: Argument \'data\' should be a dictionary.'
        data_mode = None  # data_mode can only be 'dataframe' or 'ndarray'
        data_schema = set(data.keys())
        if set(["df"]).issubset(data_schema):
            data_mode = 'dataframe'
        if set(["x", "y"]).issubset(data_schema):
            data_mode = 'ndarray'
        assert data_mode in ['dataframe', 'ndarray'],\
            'ERROR: Argument \'data\' should fit either \
                dataframe schema (include \'df\' in keys) or\
                     ndarray (include \'x\' and \'y\' in keys) schema.'

        # data extract
        if data_mode == 'dataframe':
            input_data = data['df']
            feature_cols = data.get("feature_cols", None)
            target_col = data.get("target_col", None)
            validation_data = data.get("val_df", None)
        else:
            input_data = {"x": data["x"], "y": data["y"]}
            if 'val_x' in data.keys():
                validation_data = {"x": data["val_x"], "y": data["val_y"]}
            else:
                validation_data = None

        # metric and metric's mode
        self.metric = metric
        self.mode = Evaluator.get_metric_mode(metric)

        # prepare parameters for search engine
        runtime_params = recipe.runtime_params()
        self.num_samples = runtime_params['num_samples']
        stop = dict(runtime_params)
        del stop['num_samples']

        # temp operation for reward_metric
        redundant_stop_keys = stop.keys() - {"reward_metric", "training_iteration"}
        assert len(redundant_stop_keys) == 0, \
            f"{redundant_stop_keys} is not expected in stop criteria, \
             only \"reward_metric\", \"training_iteration\" are expected."

        if "reward_metric" in stop.keys():
            stop[self.metric] = -stop["reward_metric"] if \
                self.mode == "min" else stop["reward_metric"]
            del stop["reward_metric"]
        stop.setdefault("training_iteration", 1)

        self.stopper = TrialStopper(stop=stop, metric=self.metric, mode=self.mode)

        if search_space is None:
            search_space = recipe.search_space()
        self.search_space = search_space

        self._search_alg = RayTuneSearchEngine._set_search_alg(search_alg, search_alg_params,
                                                               recipe, self.metric, self.mode)
        self._scheduler = RayTuneSearchEngine._set_scheduler(scheduler, scheduler_params,
                                                             self.metric, self.mode)

        if feature_transformers is None and data_mode == 'dataframe':
            feature_transformers = IdentityTransformer(feature_cols, target_col)

        numpy_format = True if data_mode == 'ndarray' else False

        self.train_func = self._prepare_train_func(input_data=input_data,
                                                   model_create_func=model_create_func,
                                                   feature_transformers=feature_transformers,
                                                   validation_data=validation_data,
                                                   metric=metric,
                                                   mc=mc,
                                                   remote_dir=self.remote_dir,
                                                   numpy_format=numpy_format
                                                   )

    @staticmethod
    def _set_search_alg(search_alg, search_alg_params, recipe, metric, mode):
        if search_alg:
            if not isinstance(search_alg, str):
                raise ValueError(f"search_alg should be of type str."
                                 f" Got {search_alg.__class__.__name__}")
            if search_alg_params is None:
                search_alg_params = dict()
            search_alg_params.update(dict(
                metric=metric,
                mode=mode,
            ))
            search_alg = tune.create_searcher(search_alg, **search_alg_params)
        return search_alg

    @staticmethod
    def _set_scheduler(scheduler, scheduler_params, metric, mode):
        if scheduler:
            if not isinstance(scheduler, str):
                raise ValueError(f"Scheduler should be of type str. "
                                 f"Got {scheduler.__class__.__name__}")
            if scheduler_params is None:
                scheduler_params = dict()
            scheduler_params.update(dict(
                time_attr="training_iteration",
                metric=metric,
                mode=mode,
            ))
            scheduler = tune.create_scheduler(scheduler, **scheduler_params)
        return scheduler

    def run(self):
        """
        Run trials
        :return: trials result
        """
        analysis = tune.run(
            self.train_func,
            local_dir=self.logs_dir,
            metric=self.metric,
            mode=self.mode,
            name=self.name,
            stop=self.stopper,
            config=self.search_space,
            search_alg=self._search_alg,
            num_samples=self.num_samples,
            scheduler=self._scheduler,
            resources_per_trial=self.resources_per_trial,
            verbose=1,
            reuse_actors=True
        )
        self.trials = analysis.trials

        # Visualization code for ray (leaderboard)
        logger_name = self.name if self.name else DEFAULT_LOGGER_NAME
        tf_config, tf_metric = TensorboardXLogger._ray_tune_searcher_log_adapt(analysis)

        self.logger = TensorboardXLogger(logs_dir=os.path.join(self.logs_dir,
                                                               logger_name+"_leaderboard"),
                                         name=logger_name)
        self.logger.run(tf_config, tf_metric)
        self.logger.close()

        return analysis

    def get_best_trials(self, k=1):
        """
        get a list of best k trials
        :params k: top k
        :return: trials list
        """
        sorted_trials = RayTuneSearchEngine._get_sorted_trials(self.trials,
                                                               metric=self.metric,
                                                               mode=self.mode)
        best_trials = sorted_trials[:k]
        return [self._make_trial_output(t) for t in best_trials]

    def _make_trial_output(self, trial):
        return TrialOutput(config=trial.config,
                           model_path=os.path.join(trial.logdir, trial.last_result["checkpoint"]))

    @staticmethod
    def _get_best_trial(trial_list, metric, mode):
        """Retrieve the best trial."""
        if mode == "max":
            return max(trial_list, key=lambda trial: trial.last_result.get(metric, 0))
        else:
            return min(trial_list, key=lambda trial: trial.last_result.get(metric, 0))

    @staticmethod
    def _get_sorted_trials(trial_list, metric, mode):
        return sorted(
            trial_list,
            key=lambda trial: trial.last_result.get(metric, 0),
            reverse=True if mode == "max" else False)

    @staticmethod
    def _get_best_result(trial_list, metric, mode):
        """Retrieve the last result from the best trial."""
        return {metric: RayTuneSearchEngine._get_best_trial(trial_list,
                                                            metric, mode).last_result[metric]}

    def test_run(self):
        def mock_reporter(**kwargs):
            assert self.metric in kwargs, "Did not report proper metric"
            assert "checkpoint" in kwargs, "Accidentally removed `checkpoint`?"
            raise GoodError("This works.")

        try:
            self.train_func({'out_units': 1,
                             'selected_features': ["MONTH(datetime)", "WEEKDAY(datetime)"]},
                            mock_reporter)
            # self.train_func(self.search_space, mock_reporter)

        except TypeError as e:
            print("Forgot to modify function signature?")
            raise e
        except GoodError:
            print("Works!")
            return 1
        raise Exception("Didn't call reporter...")

    @staticmethod
    def _prepare_train_func(input_data,
                            model_create_func,
                            feature_transformers,
                            metric,
                            validation_data=None,
                            mc=False,
                            remote_dir=None,
                            numpy_format=False,
                            ):
        """
        Prepare the train function for ray tune
        :param input_data: input data
        :param model_create_func: model create function
        :param feature_transformers: feature transformers
        :param metric: the rewarding metric
        :param validation_data: validation data
        :param mc: if calculate uncertainty
        :param remote_dir: checkpoint will be uploaded to remote_dir in hdfs
        :param numpy_format: whether input data is of numpy format

        :return: the train function
        """
        numpy_format_id = ray.put(numpy_format)
        input_data_id = ray.put(input_data)
        ft_id = ray.put(feature_transformers)

        # validation data processing
        df_not_empty = isinstance(validation_data, dict) or\
            (isinstance(validation_data, pd.DataFrame) and not validation_data.empty)
        df_list_not_empty = isinstance(validation_data, dict) or\
            (isinstance(validation_data, list) and validation_data
                and not all([d.empty for d in validation_data]))
        if validation_data is not None and (df_not_empty or df_list_not_empty):
            validation_data_id = ray.put(validation_data)
            is_val_valid = True
        else:
            is_val_valid = False

        def train_func(config):
            numpy_format = ray.get(numpy_format_id)

            if isinstance(model_create_func, ModelBuilder):
                trial_model = model_create_func.build(config)
            else:
                trial_model = model_create_func()

            if not numpy_format:
                global_ft = ray.get(ft_id)
                trial_ft = deepcopy(global_ft)
                imputer = None
                if "imputation" in config:
                    if config["imputation"] == "LastFillImpute":
                        imputer = LastFillImpute()
                    elif config["imputation"] == "FillZeroImpute":
                        imputer = FillZeroImpute()

                # handling input
                global_input_df = ray.get(input_data_id)
                trial_input_df = deepcopy(global_input_df)
                if imputer:
                    trial_input_df = imputer.impute(trial_input_df)
                config = convert_bayes_configs(config).copy()
                (x_train, y_train) = trial_ft.fit_transform(trial_input_df, **config)

                # handling validation data
                validation_data = None
                if is_val_valid:
                    global_validation_df = ray.get(validation_data_id)
                    trial_validation_df = deepcopy(global_validation_df)
                    validation_data = trial_ft.transform(trial_validation_df)
            else:
                train_data = ray.get(input_data_id)
                x_train, y_train = (train_data["x"], train_data["y"])
                validation_data = None
                if is_val_valid:
                    validation_data = ray.get(validation_data_id)
                    validation_data = (validation_data["x"], validation_data["y"])
                trial_ft = None

            # no need to call build since it is called the first time fit_eval is called.
            # callbacks = [TuneCallback(tune_reporter)]
            # fit model
            best_reward = None
            for i in range(1, 101):
                result = trial_model.fit_eval(data=(x_train, y_train),
                                              validation_data=validation_data,
                                              mc=mc,
                                              metric=metric,
                                              # verbose=1,
                                              **config)
                reward = result
                checkpoint_filename = "best.ckpt"

                # Save best reward iteration
                mode = Evaluator.get_metric_mode(metric)
                if mode == "max":
                    has_best_reward = best_reward is None or reward > best_reward
                else:
                    has_best_reward = best_reward is None or reward < best_reward

                if has_best_reward:
                    best_reward = reward
                    if isinstance(model_create_func, ModelBuilder):
                        trial_model.save(checkpoint_filename)
                    else:
                        save_zip(checkpoint_filename, trial_ft, trial_model, config)
                    # Save to hdfs
                    if remote_dir is not None:
                        upload_ppl_hdfs(remote_dir, checkpoint_filename)

                report_dict = {"training_iteration": i,
                               metric: reward,
                               "checkpoint": checkpoint_filename,
                               "best_" + metric: best_reward}
                tune.report(**report_dict)

        return train_func


# stopper
class TrialStopper(Stopper):
    def __init__(self, stop, metric, mode):
        self._mode = mode
        self._metric = metric
        self._stop = stop

    def __call__(self, trial_id, result):
        if self._metric in self._stop.keys():
            if self._mode == "max" and result[self._metric] >= self._stop[self._metric]:
                return True
            if self._mode == "min" and result[self._metric] <= self._stop[self._metric]:
                return True
        if "training_iteration" in self._stop.keys():
            if result["training_iteration"] >= self._stop["training_iteration"]:
                return True
        return False

    def stop_all(self):
        return False
