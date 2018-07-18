import tensorflow as tf
import matplotlib.pyplot as plt
from zoo.pipeline.api.keras.models import Model

from zoo.pipeline.api.keras.layers import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from utils import *
from zoo import init_nncontext
from zoo.pipeline.api.torch.criterion import *
import gym

EP_MAX = 1000
EP_LEN = 320
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

def build_model(state_size, action_size):
    # critic

    critic_input = Input(shape=(state_size, ))
    v_l1 = Dense(100, activation="relu")(critic_input)
    state_value = Dense(1)(v_l1)

    # actor
    actor_input = Input(shape=(state_size, ))
    a_l1 = Dense(100, activation="relu")(actor_input)

    a_l2_mu = Dense(action_size, activation="tanh")(a_l1)
    mu = MulConstant(2.0)(a_l2_mu)

    a_l2_sigma = Dense(action_size, activation="softplus")(a_l1)
    log_sigma = Log()(a_l2_sigma)

    input = Input(shape=(state_size, ))
    critic = Model([critic_input], [state_value])(input)
    actor = Model([actor_input], [mu, log_sigma])(input)

    model = Model([input], [actor, critic])

    return model


class PPO(object):

    def __init__(self, state_size, action_size, sc):
        self.sc = sc
        self.model = build_model(state_size, action_size)
        self.optimizer = None
        self.criterion = ParallelCriterion()
        self.criterion.add(CPPOCriterion(epsilon=0.2), 1.0)
        self.criterion.add(MSECriterion(), 2.0)
        self.iter = 0

    def update(self, data):
        def to_sample(x):
            return Sample.from_ndarray(x[0], [np.stack([x[1], x[2] - x[5], x[3], x[4]]), x[2]])

        rdd = self.sc.parallelize(data).map(lambda x: to_sample(x))
        if self.optimizer is None:
            self.optimizer = Optimizer.create(model=self.model,
                                              training_set=rdd,
                                              criterion=self.criterion,
                                              optim_method=Adam(learningrate=0.0001),
                                              end_trigger=MaxIteration(self.iter + 10),
                                              batch_size=BATCH)
        else:
            self.optimizer.set_traindata(rdd, BATCH)
            self.optimizer.set_end_when(MaxIteration(self.iter + 10))
        self.optimizer.optimize()
        self.iter = self.iter + 10

    def choose_action(self, s):
        s = s[np.newaxis, :]
        result = self.model.forward(s)
        mu = np.squeeze(result[0][0], 0)
        log_sigma = np.squeeze(result[0][1], 0)
        state_v = np.squeeze(result[1], 0)
        return mu, log_sigma, state_v

sc = init_nncontext()
env = gym.make('Pendulum-v0').unwrapped
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
ppo = PPO(state_size, action_size, sc)
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r, buffer_mu, buffer_log_sigma, buffer_v = [], [], [], [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        mu, log_sigma, state_v = ppo.choose_action(s)
        a = np.random.multivariate_normal(mu, np.diag(np.exp(log_sigma)))
        a = np.clip(a, -2, 2)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        buffer_mu.append(mu)
        buffer_log_sigma.append(log_sigma)
        buffer_v.append(state_v)

        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = state_v
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            data = []
            for i in range(len(buffer_s)):
                data.append((buffer_s[i], buffer_a[i], discounted_r[i], buffer_mu[i], buffer_log_sigma[i], buffer_v[i]))

            # bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            # b_mu = np.vstack(buffer_mu)
            # b_sigma = np.vstack(buffer_sigma)
            # buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(data)
            buffer_s, buffer_a, buffer_r, buffer_mu, buffer_log_sigma, buffer_v = [], [], [], [], [], []
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'SimplePPO: Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()