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

# Copyright 2017 The Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pickle

import ray

from zoo.orca.learn.tf.tf_runner import TFRunner

logger = logging.getLogger(__name__)


class TFTrainer:
    def __init__(self,
                 model_creator,
                 data_creator,
                 config=None,
                 num_replicas=1,
                 num_cpus_per_worker=1,
                 verbose=False):
        """Sets up the TensorFlow trainer.
        Args:
            model_creator (dict -> Model): This function takes in the `config`
                dict and returns a compiled TF model.
            data_creator (dict -> tf.Dataset, tf.Dataset): Creates
                the training and validation data sets using the config.
                `config` dict is passed into the function.
            config (dict): configuration passed to 'model_creator',
                'data_creator'. Also contains `fit_config`, which is passed
                into `model.fit(data, **fit_config)` and
                `evaluate_config` which is passed into `model.evaluate`.
            num_cpus_per_worker (int): Sets the cpu requirement for each
                worker.
            num_replicas (int): Sets number of workers used in distributed
                training. Workers will be placed arbitrarily across the
                cluster.
            use_gpu (bool): Enables all workers to use GPU.
            verbose (bool): Prints output of one model if true.
        """
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.config = {} if config is None else config
        self.num_cpus_per_worker = num_cpus_per_worker
        self.num_replicas = num_replicas
        self.verbose = verbose

        # Generate actor class
        # todo: are these resource quotas right?
        # should they be exposed to the client codee?
        Runner = ray.remote(
            num_cpus=self.num_cpus_per_worker)(TFRunner)

        # todo: should we warn about using
        # distributed training on one device only?
        # it's likely that whenever this happens it's a mistake
        if num_replicas == 1:
            # Start workers
            self.workers = [
                Runner.remote(
                    model_creator,
                    data_creator,
                    config=self.config,
                    verbose=self.verbose)
            ]
            # Get setup tasks in order to throw errors on failure
            ray.get(self.workers[0].setup.remote())
        else:
            # Start workers
            self.workers = [
                Runner.remote(
                    model_creator,
                    data_creator,
                    config=self.config,
                    verbose=self.verbose and i == 0)
                for i in range(num_replicas)
            ]

            # Compute URL for initializing distributed setup
            ips = ray.get(
                [worker.get_node_ip.remote() for worker in self.workers])
            ports = ray.get(
                [worker.find_free_port.remote() for worker in self.workers])

            urls = [
                "{ip}:{port}".format(ip=ips[i], port=ports[i])
                for i in range(len(self.workers))
            ]

            # Get setup tasks in order to throw errors on failure
            ray.get([
                worker.setup_distributed.remote(urls, i, len(self.workers))
                for i, worker in enumerate(self.workers)
            ])

    def train(self):
        """Runs a training epoch."""

        # see ./tf_runner.py:setup_distributed
        # for an explanation of only taking the first worker's data
        worker_stats = ray.get([w.step.remote() for w in self.workers])
        stats = worker_stats[0].copy()
        return stats

    def validate(self):
        """Evaluates the model on the validation data set."""
        logger.info("Starting validation step.")

        # see ./tf_runner.py:setup_distributed
        # for an explanation of only taking the first worker's data
        stats = ray.get([w.validate.remote() for w in self.workers])
        stats = stats[0].copy()
        return stats

    def get_model(self):
        """Returns the learned model."""
        state = ray.get(self.workers[0].get_state.remote())
        return self._get_model_from_state(state)

    def save(self, checkpoint):
        """Saves the model at the provided checkpoint.
        Args:
            checkpoint (str): Path to target checkpoint file.
        """

        state = ray.get(self.workers[0].get_state.remote())

        with open(checkpoint, "wb") as f:
            pickle.dump(state, f)

        return checkpoint

    def restore(self, checkpoint):
        """Restores the model from the provided checkpoint.
        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        with open(checkpoint, "rb") as f:
            state = pickle.load(f)

        state_id = ray.put(state)
        ray.get([worker.set_state.remote(state_id) for worker in self.workers])

    def shutdown(self):
        """Shuts down workers and releases resources."""
        for worker in self.workers:
            worker.shutdown.remote()
            worker.__ray_terminate__.remote()

    def _get_model_from_state(self, state):
        """Creates model and load weights from state"""

        model = self.model_creator(self.config)
        model.set_weights(state["weights"])
        return model