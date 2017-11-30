from game import GabrieleCirulli2048
from RL_brain import DeepQNetwork
from my_rlbrain import DqnCon
import numpy as np
import matplotlib.pyplot as plt

# train默认为0，rule默认为1
# action 0,1,2,3 left,right,up,down
# state 16 np.arrary
# RL = DeepQNetwork(n_actions=4,
#                   n_features=16,
#                   learning_rate=0.01, e_greedy=0.97,
#                   replace_target_iter=200, memory_size=50000,
#                   e_greedy_increment=0.0008, batch_size=1000)

RL = DqnCon()
env = GabrieleCirulli2048(train=1, rule=0)
totle = 0
plt.figure()
plt.ion()  # interactive mode on
plt.axis([0, 1000, 0, 2000])
for i in range(1000):
    observation = env.reset()
    playloops = 0
    while True:
        # env.render()
        action = RL.choose_action(observation)
        observation_next, reward, done = env.step(action)
        if (observation_next == observation).all():
            reward = -10
        RL.store_transition(observation, action, reward, observation_next)
        playloops += 1
        totle += 1
        if totle > 2000:
            RL.learn()
        if done:
            break
        observation = observation_next
    if i == 0:
        scole_eps = env.get_score() * np.ones(5)
    else:
        scole_eps[:-1] = scole_eps[1:]
        scole_eps[-1] = env.get_score()
    plt.scatter(i, np.mean(scole_eps))  # 画散点图
    plt.pause(0.01)
    print("第%d轮，坚持%d步,得到%d分,探索概率%.2f" % (i, playloops, env.get_score(), RL.get_epi()))
