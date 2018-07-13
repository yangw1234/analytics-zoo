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

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from agent import *
from utils import *
from environment import *
# from criterion import *
import gym
from gym import wrappers
import math

GAMMA = 0.95


def build_model(state_size, action_size):
    # critic

    # value_input = Input()
    # v_l1 = Linear(state_size, 100)(value_input)
    # a_relu1 = ReLU()(v_l1)
    # state_value = Linear(100, 1)(a_relu1)

    # actor
    actor_input = Input()
    a_l1 = Linear(state_size, 100)(actor_input)
    a_relu1 = ReLU()(a_l1)

    a_l2_mu = Linear(100, action_size)(a_relu1)
    a_tanh_mu = Tanh()(a_l2_mu)
    mu = MulConstant(2.0)(a_tanh_mu)

    a_l2_sigma = Linear(100, action_size)(a_relu1)
    sigma = SoftPlus()(a_l2_sigma)

    # input = Input()
    # critic = Model([value_input], [state_value])(input)
    actor = Model([actor_input], [mu, sigma])(input)

    # model = Model([input], [actor, critic])

    return actor


def create_agent(x):
    env = gym.make('Pendulum-v0')
    env = GymEnvWrapper(env)
    return PPOAgent(env, 498, use_gae=True)


# def calc_baseline(r_rewards):
#     max_steps = r_rewards.map(lambda x: x.shape[0]).max()
#     pad = r_rewards.map(lambda x: np.pad(x, (0, max_steps-x.shape[0]), 'constant'))
#     sum, count = pad.map(lambda x: (x, 1)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
#     mean = sum / count
#     return mean

# (obs, actions, adv, action_prob_mean, action_prob_log_std)
def to_sampe(x):

    Sample.from_ndarray(x[0], [x[1], x[2], x[3], x[4]])


def normalize(records, eps=1e-8):
    stats = records.map(lambda x: x[2]).stats()

    mean = stats.mean()
    std = stats.sampleStdev()

    return records.map(lambda x: (x[0], x[1], (x[2] - mean) / (std + eps), x[3], x[4], x[5]))


if __name__ == "__main__":


    env = gym.make('Pendulum-v0')
    env = wrappers.Monitor(env, "/tmp/cartpole-experiment", video_callable=lambda x: True, force=True)
    env = GymEnvWrapper(env)
    test_agent = PPOAgent(env, 1000, use_gae=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    model = build_model(state_size, action_size)

    test_agent.sample(model, num_steps=100)

    exit(0)

    spark_conf = create_spark_conf()

    sc = SparkContext(appName="PPO-Pendulum-v0", conf=spark_conf)
    # JavaCreator.set_creator_class("com.intel.analytics.bigdl.rl.python.api.RLPythonBigDL")
    init_engine()
    init_executor_gateway(sc)
    redire_spark_logs()
    # show_bigdl_info_logs()

    node_num, core_num = get_node_and_core_number()
    parallelism = node_num * core_num

    print "parallelism %s " % parallelism

    # test environment on driver
    env = gym.make('Pendulum-v0')
    env = wrappers.Monitor(env, "/tmp/cartpole-experiment", video_callable=lambda x: True, force=True)
    env = GymEnvWrapper(env)
    test_agent = PPOAgent(env, 1000, use_gae=False)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = build_model(state_size, action_size)
    criterion = ContinuousPPOCriterion()

    # create and cache several agents on each partition as specified by parallelism
    # and cache it
    with DistributedAgents(sc, create_agent=create_agent, parallelism=parallelism) as a:
        # a.agents is a RDD[Agent]
        agents = a.agents
        optimizer = None

        for i in range(60):
            with SampledTrajs(sc, agents, model, num_steps_per_part=200) as trajs:
                # samples is a RDD[Trajectory]
                trajs = trajs.samples \
                    .map(lambda traj: (traj.data["observations"],
                                       traj.data["actions"],
                                       traj.data["rewards"],
                                       traj.data["advantages"],
                                       traj.data["action_prob"][0],
                                       traj.data["action_prob"][1]))

                rewards_stat = trajs.map(lambda traj: traj[2].sum()).stats()

                print "*********** steps %s **************" % i
                print "reward mean:", rewards_stat.mean()
                print "reward std:", rewards_stat.sampleStdev()
                print "reward max:", rewards_stat.max()

                trajs = trajs.map(lambda x: (x[0], x[1], x[3], x[4], x[5]))

                # trajectories to records (obs, actions, adv, action_prob_mean, action_prob_log_std, value_targets)
                records = trajs.flatMap(lambda x: [(x[0][i], x[1][i], x[2][i], x[3][i], x[4][i]) for i in range(len(x[0]))])

                num_records = records.count()
                batch_size = num_records - num_records % parallelism
                # batch_size = 100

                print "total %s num_records" % num_records
                print "using %s batch_size" % batch_size

                # normalize advantages
                normalized = normalize(records)

                # to bigdl sample
                # data = normalized.map(lambda x: obs_act_adv_old_prob_to_sample(x, 2))
                data = normalized.map(lambda x: obs_act_adv_to_sample(x, 2, False))

                # update one step
                if optimizer is None:
                    optimizer = Optimizer(model=model,
                                          training_rdd=data,
                                          criterion=criterion,
                                          optim_method=RMSprop(learningrate=0.005),
                                          end_trigger=MaxIteration(1),
                                          batch_size=batch_size)
                else:
                    optimizer.set_traindata(data, batch_size)
                    optimizer.set_end_when(MaxIteration(i + 1))

                model = optimizer.optimize()

                if (i + 1) % 10 == 0:
                    import time
                    start = time.time()
                    step = test_agent.sample(model, num_trajs=1)[0].data["actions"].shape[0]
                    end = time.time()
                    print "************************************************************************"
                    print "*****************sample video generated, %s steps, using %s seconds**********************" % (step, start - end)
                    print "************************************************************************"

    env.gym.close()