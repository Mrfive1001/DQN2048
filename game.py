#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Gabriele Cirulli's 2048 puzzle game.

    Python3/tkinter port by Raphaël Seban <motus@laposte.net>

    Copyright (c) 2014+ Raphaël Seban for the present code.

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.

    If not, see http://www.gnu.org/licenses/
"""

import random
import copy
import numpy as np
import time
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD

try:
    import tkinter as tk
    import ttk
    import tkMessageBox as messagebox
except:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
# end try

from src import game2048_score as GS
from src import game2048_grid as GG
from RL_brain import DeepQNetwork

RL = DeepQNetwork(n_actions=4,
                  n_features=16,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=200, memory_size=2000,
                  e_greedy_increment=0.0008, )


class GabrieleCirulli2048(tk.Tk):
    PADDING = 10  # 控制界面的
    START_TILES = 2  # 初始化几个方格

    def __init__(self, **kw):
        tk.Tk.__init__(self)

        self.train = kw.get("train", 0)  # 从类读取是否训练
        self.playloops = 0  # 玩的次数
        self.ran = 1  # 初始化记忆池

        self.ai_time = 100  # 电脑玩的时候每隔多久执行一次
        self.memory = []  # 记忆池
        self.memory_size = 0  # 记忆池容量
        self.gamma = 0.3  # 长远收益的比例
        self.episi = 0.2  # 探索概率
        self.old_state_action = []
        self.oldst = []
        self.epoch = 0  # 神经网络训练轮数

        # 真实Q网络
        self.real = Sequential()

        # 目标Q网络
        self.target = Sequential()

        self.score_list = []  # 得到的分数列表
        self.initialize(**kw)  # 画图的初始化

    def run(self, **kw):
        # 主程序的执行过程，如果训练那就不画图了
        if self.train:
            self.ai_train()
        else:
            # 画出游戏界面
            self.center_window()
            self.deiconify()
            self.new_game(**kw)
            self.mainloop()

    # 画图
    def center_window(self, tk_event=None, *args, **kw):
        self.update_idletasks()
        _width = self.winfo_reqwidth()
        _height = self.winfo_reqheight()
        _screen_width = self.winfo_screenwidth()
        _screen_height = self.winfo_screenheight()
        _left = (_screen_width - _width) // 2
        _top = (_screen_height - _height) // 2
        self.geometry("+{x}+{y}".format(x=_left, y=_top))

    # 图形初始化
    def initialize(self, **kw):
        self.title("2048")  # 标题
        self.protocol("WM_DELETE_WINDOW", self.quit_app)  # 退出协议
        self.resizable(width=False, height=False)  # 设置不可调
        self.withdraw()
        ttk.Style().configure(".", font="sans 10")  # 样式
        _pad = self.PADDING
        self.hint = ttk.Label(
            self, text="Hint: use keyboard arrows to move tiles."
        )
        self.grid = GG.Game2048Grid(self, **kw)  # 窗格
        self.score = GS.Game2048Score(self, **kw)
        self.hiscore = GS.Game2048Score(self, label="Highest:", **kw)
        self.grid.pack(side=tk.TOP, padx=_pad, pady=_pad)
        self.hint.pack(side=tk.TOP)
        self.score.pack(side=tk.LEFT)
        self.hiscore.pack(side=tk.LEFT)
        ttk.Button(
            self, text="Quit!", command=self.quit_app,
        ).pack(side=tk.RIGHT, padx=_pad, pady=_pad)
        ttk.Button(
            self, text="New Game", command=self.new_game,
        ).pack(side=tk.RIGHT)
        ttk.Button(
            self, text="AI Game", command=self.ai_new_game,
        ).pack(side=tk.RIGHT)
        self.grid.set_score_callback(self.update_score)

        # define your AI variable here

    # 建立新游戏
    def new_game(self, *args, **kw):
        self.unbind_all("<Key>")
        self.score.reset_score()
        self.grid.reset_grid()
        for n in range(self.START_TILES):
            self.after(
                100 * random.randrange(3, 7), self.grid.pop_tile
            )
        self.bind_all("<Key>", self.on_keypressed)

    # 退出功能
    def quit_app(self, **kw):
        if messagebox.askokcancel("Question", "Quit game?"):
            self.quit()
            self.destroy()

    # 摁键功能描述
    def on_keypressed(self, tk_event=None, *args, **kw):
        _event_handler = {
            "left": self.grid.move_tiles_left,
            "right": self.grid.move_tiles_right,
            "up": self.grid.move_tiles_up,
            "down": self.grid.move_tiles_down,
            "escape": self.quit_app,
        }.get(tk_event.keysym.lower())
        try:
            _event_handler()
            self.hint.pack_forget()
        except:
            pass

        tiles = self.grid.tiles  # 包含的窗格的地方和值
        # print("tiles = {}".format(tiles))
        for t in tiles:
            print("Tile id = {}, tile row = {}, tile column = {}, value = {}".
                  format(t, tiles[t].row, tiles[t].column, tiles[t].value))
        print("--------------------------")
        # end try

    # 更新分数
    def update_score(self, value, mode="add"):
        if str(mode).lower() in ("add", "inc", "+"):
            self.score.add_score(value)
        else:
            self.score.set_score(value)

        self.hiscore.high_score(self.score.get_score())

    # 界面新开一个ai游戏
    def ai_new_game(self, *args, **kw):
        self.unbind_all("<Key>")
        self.score.reset_score()
        self.grid.reset_grid()
        for n in range(self.START_TILES):
            self.after(
                10 * random.randrange(3, 7), self.grid.pop_tile
            )
        self.playloops = 0
        self.after(self.ai_time, self.ai_pressed)  # 多长时间后调用下一次ai_pressed
        self.bind_all("<Key>", self.on_keypressed)

    # 定义一个AI程序，按了界面上的ai运行按钮后会定时触发
    # 在这个子程序里面运行一次AI操作
    # 不需要训练的在界面显示的ai展示
    def ai_pressed(self, tk_event=None, *args, **kw):
        matrix = self.grid.matrix.matrix
        self.playloops = self.playloops + 1
        mat2048 = np.zeros((4, 4))
        tiles = self.grid.tiles
        for t in tiles:
            mat2048[tiles[t].row, tiles[t].column] = np.log2(tiles[t].value)
        pressed = self.ai_move(mat2048)  # this is random control
        if pressed == 1:
            print("Move left\n")
            self.grid.move_tiles_left()
        elif pressed == 2:
            print("Move right\n")
            self.grid.move_tiles_right()
        elif pressed == 3:
            print("Move up\n")
            self.grid.move_tiles_up()
        elif pressed == 4:
            print("Move down\n")
            self.grid.move_tiles_down()
        else:
            pass
        if self.grid.no_more_hints():  # game over
            # self.ai_new_game()  # play ai again
            pass
        else:
            self.after(self.ai_time, self.ai_pressed)  # ai press again after 200 ms

    # 生成记忆池
    def save_memory(self, memo_size=5000):
        self.memory = []
        while self.memory_size <= memo_size:
            # 开始游戏前的准备
            self.playloops = 0
            self.score.reset_score()
            self.grid.clear_all()
            # 随机加两个数字
            for n in range(self.START_TILES):
                self.grid.pop_tile()  # 对象加1
            # 当没有结束的时候，一直循环
            while not self.grid.no_more_hints():  # game over
                mat_sta, action, reward = self.ai_transfer()
                if self.playloops == 1:
                    self.old_state_action = [mat_sta, action, reward]
                else:
                    self.memory_size += 1
                    print("已收集", self.memory_size, "条数据进记忆池")
                    # 加入记忆[[state,action,reward],next_state,not_end]
                    self.memory.append([self.old_state_action, mat_sta])
                    # 判断状态是否结束
                    if self.grid.no_more_hints():
                        self.memory[-1].append(0)
                    else:
                        self.memory[-1].append(1)
                    self.old_state_action = [mat_sta, action, reward]
        print("记忆池添加结束！")

    def add_memory(self, mar):
        # 与保存记忆池差不太多
        # mar = (mstate,action,reward)
        mat_sta, action, reward = mar
        if self.playloops == 1:
            self.old_state_action = mar
        else:
            self.memory.pop(random.randint(0, self.memory_size))
            # newq = self.real.predict(mat_sta.reshape((1, -1))).reshape(4)
            # self.memory[-1][1][self.old_state_action[1] - 1] \
            #     = self.old_state_action[2] + newq.max()
            # # q = r + gamma * q'
            self.memory.append([self.old_state_action, mat_sta])
            if self.grid.no_more_hints():
                self.memory[-1].append(0)
            else:
                self.memory[-1].append(1)
            self.old_state_action = [mat_sta, action, reward]

    def nn_init(self):
        self.real.add(Dense(input_dim=16, activation="relu", units=500))
        self.real.add(Dense(units=500, activation='relu'))
        self.real.add(Dense(units=4, activation='linear'))
        self.real.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

        self.target.add(Dense(input_dim=16, activation="relu", units=500))
        self.target.add(Dense(units=500, activation='relu'))
        self.target.add(Dense(units=4, activation='linear'))
        # self.real.fit(x_train[:200, :], y_train[:200, :], batch_size=10, initial_epoch=20, epochs=40)

    def update_nn(self, size=100):
        # 从记忆池里面随机选出size个样本
        x_train, y_train = [], []
        for i in np.random.choice(range(self.memory_size), size):
            value = self.memory[i]
            s1, a1, r1 = value[0]
            s2 = value[1]
            q1_old = self.real.predict(s1.reshape((1, -1))).reshape(4)
            q1_new = q1_old
            if value[2] == 0:  # 结束了
                q1_new[a1 - 1] = self.score.get_score()
            else:  # 更新Q值
                q2 = self.target.predict(s2.reshape((1, -1))).reshape(4)
                q1_new[a1 - 1] = 1 + self.gamma * np.max(q2)
            x_train.append(s1)
            y_train.append(q1_new)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.real.fit(x_train, y_train, batch_size=int(size / 2),
                      initial_epoch=self.epoch, epochs=self.epoch + 10, verbose=0)  # 不加显示
        self.epoch += 10

    def update_target(self):
        self.real.save_weights('m1.h5')
        self.target.load_weights('m1.h5')

    # 修改这个子程序
    def ai_move(self, mat2048):
        # 输入是2048表格的2对数，输出1~4，表示上下左右 np.array((16,))
        # return random.randint(1, 4)
        if self.ran or np.random.binomial(1, self.episi):  # 1出现episi的概率
            return random.randint(1, 4)
        else:
            return np.argmax(self.real.predict(mat2048.reshape((1, -1))).reshape(4)) + 1

    def ai_transfer(self):
        # 可以返回状态、动作和奖励
        self.playloops = self.playloops + 1
        mat2048 = np.zeros(16)
        tiles = self.grid.tiles
        for t in tiles:
            mat2048[tiles[t].row * 4 + tiles[t].column] = np.log2(tiles[t].value)
        # 2048表格的2对数
        mat2048 = mat2048.astype(int)
        old = self.score.get_score()
        # pressed = self.ai_move(mat2048)  # this is random control
        pressed = RL.choose_action(mat2048)
        self.oldst = [mat2048, pressed]
        if pressed == 0:
            self.grid.move_tiles_left()
        elif pressed == 1:
            self.grid.move_tiles_right()
        elif pressed == 2:
            self.grid.move_tiles_up()
        elif pressed == 3:
            self.grid.move_tiles_down()
        else:
            pass
        new = self.score.get_score()  # 读取总分数
        return mat2048, pressed, new - old

    def ai_train(self):
        # self.save_memory()  # 增加记忆池
        # self.nn_init()  # 初始化nn
        # self.update_target()  # 更新target网络
        # self.ran = 0
        totle_step = 0
        for item in range(5000):
            # 初始化
            self.playloops = 0
            self.score.reset_score()
            self.grid.clear_all()
            for n in range(self.START_TILES):
                self.grid.pop_tile()  # 对象加1
            while not self.grid.no_more_hints():  # game over
                mat_sta, action, reward = self.ai_transfer()
                RL.store_transition(self.oldst[0], self.oldst[1],
                                    reward, mat_sta)

                totle_step += 1
                if totle_step % 50 == 0:
                    RL.learn()
                # if self.playloops == 1:
                #     self.old_state_action = [mat_sta, action, reward]
                #     continue
                # else:
                #     RL.store_transition(self.old_state_action[0], self.old_state_action[1]
                #                         )
                # self.add_memory(mar)  # 添加记忆
                # print("第%d步" % self.playloops)
                # if self.playloops % 1 == 0:
            #     self.update_nn()  # 更新real nn
            #     if self.playloops % 20 == 0:  # 20步更新一次target
            #         self.update_target()
            # self.update_target()
            self.score_list.append(self.score.get_score())
            print("第%d轮，分数是%d" % (item + 1, self.score.get_score()))
        # 保存结果
        with open('myscore.pkl', 'wb') as f:
            pickle.dump(self.score_list, f)


if __name__ == "__main__":
    GabrieleCirulli2048(train=1).run()
