from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from agent import *
from utils import *
from environment import *
import gym
from gym import wrappers
import math
from criterion import *

GAMMA = 0.95

from bigdl.util.common import JavaCreator
JavaCreator.set_creator_class("com.intel.analytics.bigdl.rl.python.api.RLPythonBigDL")



if __name__ == "__main__":
    spark_conf = create_spark_conf()
    sc = SparkContext(appName="REINFORCE_CartPole-v1", conf=spark_conf)
    init_engine()

    criterion = ContinuousPPOCriterion()

    mean = np.array([[1.0, 2.0, 3.0]])

    log_std = np.array([[0.5, 0.4, 0.3]])

    actions = np.array([[1.5, 1.8, 3.1]])
    advantage = np.array([10.0])
    old_mean = np.array([[0.8, 1.8, 2.8]])
    old_log_std = np.array([[0.6, 0.3, 0.2]])

    loss = criterion.forward([mean, log_std], [actions, advantage, old_mean, old_log_std])

    grad_mean, grad_log_std = criterion.backward([mean, log_std], [actions, advantage, old_mean, old_log_std])
    print loss
    print grad_mean
    print grad_log_std