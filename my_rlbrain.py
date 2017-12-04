from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
import numpy as np
import move


class DqnCon:
    def __init__(self, **kw):
        # s为一维向量
        self.s_shape = kw.get('s_shape', 16)
        self.a_num = kw.get('a_shape', 4)
        self.memory_size = 2000
        self.memory = np.zeros((self.memory_size, self.s_shape * 2 + 3))
        self.memory_counter = 0
        self.episilon_min = 0.05
        self.episilon = 0.1
        self.gamma = 0.90
        self.observation = None
        self.observation_next = None
        self.playloops = 0
        self.update_time = 0
        self.nl = 120
        self.act = 'relu'
        self.e_decrease = 0.0008
        # 真实Q网络
        self.real = Sequential()
        self.real.add(Conv2D(10, (3, 3), strides=(1, 1), padding='same', activation=self.act,
                             input_shape=(4, 4, 1), data_format='channels_last'))
        self.real.add(Activation('relu'))
        self.real.add(Conv2D(10, (3, 3), strides=(1, 1), padding='same', activation=self.act))
        self.real.add(Activation('relu'))
        self.real.add(Flatten())
        self.real.add(Dense(activation=self.act, units=self.nl))
        self.real.add(Dense(activation=self.act, units=self.nl))
        self.real.add(Dense(units=self.a_num, activation='relu'))
        self.real.compile(loss='mse', optimizer=RMSprop(lr=0.01), )
        # 目标Q网络
        self.target = Sequential()
        self.target.add(Conv2D(10, (3, 3), strides=(1, 1), padding='same', activation=self.act,
                               input_shape=(4, 4, 1), data_format='channels_last'))
        self.target.add(Activation('relu'))
        self.target.add(Conv2D(10, (3, 3), strides=(1, 1), padding='same', activation=self.act))
        self.target.add(Activation('relu'))
        self.target.add(Flatten())
        self.target.add(Dense(activation=self.act, units=self.nl))
        self.target.add(Dense(activation=self.act, units=self.nl))
        self.target.add(Dense(units=self.a_num, activation='relu'))
        self.target.compile(loss='mse', optimizer=RMSprop(lr=0.01), )

    def update_target(self):
        self.target.set_weights(self.real.get_weights())

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, [a, r], s_, done))  # 将一维的组合成一个一维的
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :].copy()  # 变成了二维行向量
        if np.random.uniform() < self.episilon:  # 1出现episi的概率
            return np.random.randint(self.a_num)
        else:
            if observation.shape != (4, 4):
                observation = observation.reshape((4, 4))
            act = self.ai_rule(observation)
            observation = observation.reshape((-1, 4, 4, 1))
            cal = self.real.predict(observation)
            q_now = act/10 + cal
            return np.argmax(q_now)

    def learn(self, size=50):
        if self.update_time % 100 == 0:
            print("更新target网络")
            self.update_target()
        sample_index = np.random.choice(min(self.memory_size, self.memory_counter), size=size)
        batch_memory = self.memory[sample_index, :]
        observation, action, reward, observation_next, done = batch_memory[:, :self.s_shape], batch_memory[:,
                                                                                              self.s_shape], \
                                                              batch_memory[:, self.s_shape + 1], batch_memory[:,
                                                                                                 -self.s_shape - 1:-1], batch_memory[
                                                                                                                        :,
                                                                                                                        -1]
        observation = observation.reshape((-1, 4, 4, 1))
        observation_next = observation_next.reshape((-1, 4, 4, 1))

        batch_state = observation
        batch_state_next = observation_next
        batch_action = action.astype(int)
        batch_reward = reward
        batch_done = done.astype(int)

        q_target = self.real.predict(batch_state)
        q_next1 = self.real.predict(batch_state_next)
        q_next2 = self.target.predict(batch_state_next)
        batch_action_withMaxQ = np.argmax(q_next1, axis=1)
        batch_index11 = np.arange(size, dtype=np.int32)
        q_next_Max = q_next2[batch_index11, batch_action_withMaxQ]
        q_target[batch_index11, batch_action] = batch_reward + (1 - batch_done) * self.gamma * q_next_Max
        self.real.fit(batch_state, q_target, verbose=0)
        self.real.fit(batch_state, q_target, verbose=0)
        # q1_old = self.real.predict(observation)
        # q1_new = q1_old.copy()
        # q2 = self.target.predict(observation_next)
        # batch_index = np.arange(size, dtype=np.int32)
        # action_index = action.astype(int)
        # q1_new[batch_index, action_index] = reward + self.gamma * np.max(q2, axis=1)
        # for j in range(self.a_num):
        #     q1_new[done == 1, j] = reward[done == 1]
        # # self.real.fit(observation, q1_new, initial_epoch=self.epoch, epochs=self.epoch + 10, verbose=0)  # 不加显示
        # self.real.fit(observation, q1_new, verbose=0)  # 不加显示
        self.update_time += 1
        self.episilon = self.episilon - self.e_decrease if self.episilon > self.episilon_min else self.episilon_min

    def get_epi(self):
        return self.episilon

    def ai_rule(self, mate):
        mat = mate.copy()  # 对数形式
        mat = 2 ** mat
        mat[mat == 1] = 0
        if mat.shape != (4, 4):
            mat = mat.reshape((4, 4))
        next_ = [move.LeftAction(mat).handleData(), move.RightAction(mat).handleData(),
                 move.UpAction(mat).handleData(), move.DownAction(mat).handleData()]
        sco = []
        for st in next_:
            if (st == mat).all():
                sco.append(-10)
            else:
                st_ = move.TestScore(st)
                sco_ = st_.EmptyTest() + st_.Monotonicity() + st_.ALLnum() + st_.equall() + st_.wheremax()
                sco.append(sco_)
        pp = np.array(sco)
        # return np.array(sco)
        mat[mat == 1] = 0
        # mat = np.log2(mat)
        return pp
