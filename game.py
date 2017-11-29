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
                  learning_rate=0.01, e_greedy=0.97,
                  replace_target_iter=100, memory_size=50000,
                  e_greedy_increment=0.000008, batch_size=500)
Size = 4


class UpdateNew(object):
    def __init__(self, matrix):
        super(UpdateNew, self).__init__()
        self.matrix = matrix
        self.score = 0
        self.zerolist = []

    def combineList(self, rowlist):
        start_num = 0
        end_num = Size - rowlist.count(0) - 1
        while start_num < end_num:
            if rowlist[start_num] == rowlist[start_num + 1]:
                rowlist[start_num] *= 2
                self.score += int(rowlist[start_num])  # 每次返回累加的分数
                rowlist[start_num + 1:] = rowlist[start_num + 2:]
                rowlist.append(0)
            start_num += 1
        return rowlist

    def removeZero(self, rowlist):
        while True:
            mid = rowlist[:]  # 拷贝一份list
            try:
                rowlist.remove(0)
                rowlist.append(0)
            except:
                pass
            if rowlist == mid:
                break
        return self.combineList(rowlist)

    def toSequence(self, matrix):
        lastmatrix = matrix.copy()
        m, n = matrix.shape  # 获得矩阵的行，列
        for i in range(m):
            newList = self.removeZero(list(matrix[i]))
            matrix[i] = newList
            for k in range(Size - 1, Size - newList.count(0) - 1, -1):  # 添加所有有0的行号列号
                self.zerolist.append((i, k))
        return matrix


class LeftAction(UpdateNew):
    def __init__(self, matrix):
        super(LeftAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy()  # 获得一份矩阵的复制
        newmatrix = self.toSequence(matrix)
        return newmatrix


class RightAction(UpdateNew):
    def __init__(self, matrix):
        super(RightAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy()[:, ::-1]
        newmatrix = self.toSequence(matrix)
        return newmatrix[:, ::-1]


class UpAction(UpdateNew):
    """docstring for UpAction"""

    def __init__(self, matrix):
        super(UpAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy().T
        newmatrix = self.toSequence(matrix)
        return newmatrix.T


class DownAction(UpdateNew):
    def __init__(self, matrix):
        super(DownAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy()[::-1].T
        newmatrix = self.toSequence(matrix)
        return newmatrix.T[::-1]


class GabrieleCirulli2048(tk.Tk):
    PADDING = 10  # 控制界面的
    START_TILES = 2  # 初始化几个方格

    def __init__(self, **kw):
        tk.Tk.__init__(self)

        self.train = kw.get("train", 0)  # 从类读取是否训练
        self.playloops = 0  # 玩的次数
        self.oldst = []
        self.score_list = []  # 得到的分数列表
        self.ai_time = 100
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
        pressed = RL.choose_action(mat2048.reshape(16))  # this is random control
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

    def ai_transfer(self):
        # 可以返回状态、动作和奖励
        self.playloops = self.playloops + 1
        mat2048 = np.zeros(16)
        tiles = self.grid.tiles
        for t in tiles:
            mat2048[tiles[t].row * 4 + tiles[t].column] = np.log2(tiles[t].value)
        # 2048表格的2对数
        mat2048 = mat2048.astype(int)
        # # print(mat2048)
        # seo = RightAction(mat2048.reshape((4, 4))).zerolist
        # print(len(seo))
        old = self.score.get_score()
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
        totle_step = 0
        for item in range(2000):
            # 初始化
            self.playloops = 0
            self.score.reset_score()
            self.grid.clear_all()
            for n in range(self.START_TILES):
                self.grid.pop_tile()  # 对象加1
            while not self.grid.no_more_hints():  # game over
                mat_sta, action, reward = self.ai_transfer()
                if self.grid.no_more_hints():
                    reward = self.score.get_score()
                elif mat_sta.all() == self.oldst[0].all():
                    reward = -10
                else:
                    reward = 1
                RL.store_transition(self.oldst[0], self.oldst[1],
                                    reward, mat_sta)
                totle_step += 1
                if totle_step > RL.memory_size:
                    # totle_step = 0
                    RL.learn()
            self.score_list.append(self.score.get_score())
            print("第%d轮，分数是%d" % (item + 1, self.score.get_score()))
        # 保存结果
        RL.plot_cost()
        with open('myscore.pkl', 'wb') as f:
            pickle.dump(self.score_list, f)


if __name__ == "__main__":
    GabrieleCirulli2048(train=1).run()
