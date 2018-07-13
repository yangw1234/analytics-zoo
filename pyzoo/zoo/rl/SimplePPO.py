import tensorflow as tf
import matplotlib.pyplot as plt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from utils import *
from zoo import init_nncontext
from zoo.pipeline.api.torch.criterion import *
import gym

EP_MAX = 1000
EP_LEN = 200
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

    critic_input = Input()
    v_l1 = Linear(state_size, 100)(critic_input)
    a_relu1 = ReLU()(v_l1)
    state_value = Linear(100, 1)(a_relu1)

    # actor
    actor_input = Input()
    a_l1 = Linear(state_size, 100)(actor_input)
    a_relu1 = ReLU()(a_l1)

    a_l2_mu = Linear(100, action_size)(a_relu1)
    a_tanh_mu = Tanh()(a_l2_mu)
    mu = MulConstant(2.0)(a_tanh_mu)

    a_l2_sigma = Linear(100, action_size)(a_relu1)
    sigma = SoftPlus()(a_l2_sigma)

    input = Input()
    critic = Model([critic_input], [state_value])(input)
    actor = Model([actor_input], [mu, sigma])(input)

    model = Model([input], [actor, critic])

    return model


class PPO(object):

    def __init__(self, state_size, action_size, sc):
        self.sc = sc
        self.model = build_model(state_size, action_size)
        self.predict_action_mu = Sequential()
        self.predict_action_mu.add(self.model)
        self.predict_action_mu.add(SelectTable(1))
        self.predict_action_mu.add(SelectTable(1))

        self.predict_action_sigma = Sequential()
        self.predict_action_sigma.add(self.model)
        self.predict_action_sigma.add(SelectTable(1))
        self.predict_action_sigma.add(SelectTable(2))

        self.predict_value = Sequential()
        self.predict_value.add(self.model)
        self.predict_value.add(SelectTable(2))
        self.optimizer = None
        self.criterion = ParallelCriterion()
        self.criterion.add(ContinuousPPOCriterion(), 1.0)
        self.criterion.add(MSECriterion(), 1.0)

    def update(self, data):

        def to_sample(x):
            return Sample.from_ndarray(x[0], [[x[1], x[2], x[3], x[4]], x[2]])

        rdd = self.sc.parallelize(data).map(lambda x: to_sample(x))
        self.optimizer = Optimizer.create(model=self.model,
                                          training_set=rdd,
                                          criterion=self.criterion,
                                          optim_method=RMSprop(learningrate=0.005),
                                          end_trigger=MaxIteration(10),
                                          batch_size=BATCH)
        self.model = self.optimizer.optimize()

    def choose_action(self, s):
        s = s[np.newaxis, :]
        mu = self.predict_action_mu.predict(s)
        mu = np.squeeze(mu, 0)
        sigma = self.predict_action_sigma.predict(s)
        sigma = np.squeeze(sigma, 0)
        state_v = self.predict_value.predict(s)
        state_v = np.squeeze(state_v, 0)
        return mu, sigma, state_v

sc = init_nncontext()
env = gym.make('Pendulum-v0').unwrapped
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
ppo = PPO(state_size, action_size, sc)
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r, buffer_mu, buffer_sigma = [], [], [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        mu, sigma, state_v = ppo.choose_action(s)
        a = np.random.multivariate_normal(mu, np.diag(sigma))
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        buffer_mu.append(mu)
        buffer_sigma.append(sigma)

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
                data.append((buffer_s[i], buffer_a[i], discounted_r[i], buffer_mu[i], buffer_sigma[i]))

            # bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            # b_mu = np.vstack(buffer_mu)
            # b_sigma = np.vstack(buffer_sigma)
            # buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(data)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()