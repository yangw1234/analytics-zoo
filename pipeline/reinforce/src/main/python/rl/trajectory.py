#
# Copyright 2016 The BigDL Authors.
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
import numpy as np

class Trajectory(object):

    fields = ["observations", "actions", "rewards", "terminal", "action_prob"]

    def __init__(self):
        self.data = {k: [] for k in self.fields}
        self.last_r = 0.0


    def add(self, **kwargs):
        '''
        add a single step to this trajectory,
        e.g. traj.add(observations=obs, actions=action, rewards=reward, terminal=terminal)
        '''
        for k, v in kwargs.items():
            self.data[k] += [v]

    def is_terminal(self):
        return self.data["terminal"][-1]

class Sampler(object):
    '''
    Helper class to sample one trajectory from the environment
    using the given model (policy or value_func/q_func estimator)
    '''

    def get_data(self, model, max_steps):
        '''
        Sample one trajectory from the environment, using the given model
        to `max_steps` steps
        '''
        raise NotImplementedError


class PolicySampler(Sampler):
    '''
    Helper class to sample one trajectory from the environment
    using the given policy
    '''

    def __init__(self, env, horizon=None, use_gae=False):
        self.horizon = horizon
        self.env = env
        self.discrete = env.discrete_action_space
        self.use_gae = use_gae
        self.last_obs = env.reset()

    def get_data(self, policy, max_steps):
        return self._run_policy(
            self.env, policy, max_steps, self.horizon)

    def _run_policy(self, env, policy, max_steps, horizon):
        length = 0

        traj = Trajectory()
        terminal = False
        for _ in range(max_steps):
            if self.use_gae:
                 output = policy.forward(self.last_obs)
                 action_distribution = output[0]
                 v_pred = output[1]
            else:
                action_distribution = policy.forward(self.last_obs)

            if self.discrete:
                action = np.random.multinomial(1, action_distribution).argmax()
            else:
                mu = action_distribution[0]
                log_sigma = action_distribution[1]
                action = np.random.multivariate_normal(mean=mu, cov=np.diag(np.square(np.exp(log_sigma))))

            observation, reward, terminal, info = env.step(action)
            action_prob = action_distribution

            length += 1
            if length >= horizon:
                terminal = True

            # Collect the experience.
            if self.use_gae:
                traj.add(observations=self.last_obs,
                         actions=action,
                         rewards=reward,
                         terminal=terminal,
                         action_prob=action_prob,
                         v_pred=v_pred)
            else:
                traj.add(observations=self.last_obs,
                         actions=action,
                         rewards=reward,
                         terminal=terminal,
                         action_prob=action_prob)

            self.last_obs = observation

            if terminal:
                self.last_obs = env.reset()
                break
        if (not terminal) and self.use_gae:
            _, traj.last_r = policy.forward(self.last_obs)

        return traj
