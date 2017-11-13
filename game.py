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


class GabrieleCirulli2048(tk.Tk):
    PADDING = 10  # 控制界面的
    START_TILES = 2  # 初始化几个方格

    def __init__(self, **kw):
        tk.Tk.__init__(self)
        self.initialize(**kw)
        self.ai_time = 100
        self.train = 0
        # self.nn = NeuralNetwork(
        #     16, 16, 4)
        # self.nn.inspect()
        # self.nnoutput = np.zeros(4)

    # end def

    def center_window(self, tk_event=None, *args, **kw):
        self.update_idletasks()
        _width = self.winfo_reqwidth()
        _height = self.winfo_reqheight()
        _screen_width = self.winfo_screenwidth()
        _screen_height = self.winfo_screenheight()
        _left = (_screen_width - _width) // 2
        _top = (_screen_height - _height) // 2
        self.geometry("+{x}+{y}".format(x=_left, y=_top))

    # end def

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
        self.grid = GG.Game2048Grid(self, **kw)   # 窗格
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

    # end def

    def new_game(self, *args, **kw):
        self.unbind_all("<Key>")
        self.score.reset_score()
        self.grid.reset_grid()
        for n in range(self.START_TILES):
            self.after(
                100 * random.randrange(3, 7), self.grid.pop_tile
            )
        # end if
        self.bind_all("<Key>", self.on_keypressed)

    # end def

    def quit_app(self, **kw):
        if messagebox.askokcancel("Question", "Quit game?"):
            self.quit()
            self.destroy()
            # end if

    # end def

    def run(self, **kw):
        self.ai_train()
        self.center_window()
        self.deiconify()
        self.new_game(**kw)
        self.mainloop()

    # end def

    def on_keypressed(self, tk_event=None, *args, **kw):
        # old = self.score.get_score()
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
        # new = self.score.get_score() # 读取总分数
        # print(new - old)
        # end try

    # end def

    def update_score(self, value, mode="add"):
        if str(mode).lower() in ("add", "inc", "+"):
            self.score.add_score(value)
        else:
            self.score.set_score(value)
        # end if
        self.hiscore.high_score(self.score.get_score())

    # end def

    def ai_new_game(self, *args, **kw):
        self.unbind_all("<Key>")
        self.score.reset_score()
        self.grid.reset_grid()
        for n in range(self.START_TILES):
            self.after(
                10 * random.randrange(3, 7), self.grid.pop_tile
            )
        # end if
        self.playloops = 0
        self.after(self.ai_time, self.ai_pressed)  # 多长时间后调用下一次ai_pressed
        self.bind_all("<Key>", self.on_keypressed)

    # end def

    # 定义一个AI程序，按了界面上的ai运行按钮后会定时触发
    # 在这个子程序里面运行一次AI操作
    def ai_pressed(self, tk_event=None, *args, **kw):
        if not self.train:
            matrix = self.grid.matrix.matrix
        # get the values of cells
        self.playloops = self.playloops + 1
        mat2048 = np.zeros((4, 4))
        tiles = self.grid.tiles
        for t in tiles:
            # put values into a matrix
            mat2048[tiles[t].row, tiles[t].column] = np.log2(tiles[t].value)
            # print("Tile id = {}, tile row = {}, tile column = {}, value = {}".
            #       format(t, tiles[t].row, tiles[t].column, tiles[t].value))
        # print(mat2048)
        # print("--------------------------")
        # add your AI program here to control the game
        # the control input is a number from 1-4
        # 1 move to left
        # 2 move to right
        # 3 move to up
        # 4 move to down
        # pressed = int(random.choice((1, 2, 3, 4)))
        pressed = self.ai_move(mat2048)  # this is random control
        if self.playloops == 1 and self.train:
            start = time.clock()
        if pressed == 1:
            if not self.train:
                print("Move left\n")
            self.grid.move_tiles_left()
        elif pressed == 2:
            if not self.train:
                print("Move right\n")
            self.grid.move_tiles_right()
        elif pressed == 3:
            if not self.train:
                print("Move up\n")
            self.grid.move_tiles_up()
        elif pressed == 4:
            if not self.train:
                print("Move down\n")
            self.grid.move_tiles_down()
        else:
            pass
        if self.playloops == 1 and self.train:
            end = time.clock()
            print("时间：", (end - start) * 100, 's')
        if self.grid.no_more_hints():  # game over
            # self.ai_new_game()  # play ai again
            pass
        else:
            if not self.train:
                self.after(self.ai_time, self.ai_pressed)  # ai press again after 200 ms
            else:
                self.ai_pressed()

    # 修改这个子程序
    def ai_move(self, mat2048):
        # 输入是2048表格的2对数，输出1~4，表示上下左右
        return random.randint(1, 4)

    # def ai_move(self, mat2048):
    #     # mat2048 是4X4的矩阵，放着的是棋盘的数据
    #     move = 0
    #     imax = mat2048.argmax()
    #     imaxrow = int(imax / 4)
    #     imaxcol = imax - imaxrow * 4
    #
    #     eq_col = np.zeros(4)  # 获得是否列中存在合并的数，如果没有，这一列就等于0
    #
    #     for i in range(4):
    #         st = []
    #         for j in range(4):  # 判断两个相邻不等于0的数是否相等，相等表示何以和
    #             if len(st) == 0:
    #                 if mat2048[j, i] > 0:
    #                     st.append(mat2048[j, i])
    #                 else:
    #                     pass
    #             else:
    #                 if mat2048[j, i] > 0:
    #                     if st.pop() == mat2048[j, i]:
    #                         eq_col[i] += mat2048[j, i]
    #                     else:
    #                         st.append(mat2048[j, i])
    #                 else:
    #                     pass
    #                     # print(st)
    #     print(eq_col)
    #
    #     eq_row = np.zeros(4)  # 获得是否行中存在合并的数，如果没有，这一行就等于0
    #     for i in range(4):
    #         st = []
    #         for j in range(4):
    #             if len(st) == 0:
    #                 if mat2048[i, j] > 0:
    #                     st.append(mat2048[i, j])
    #                 else:
    #                     pass
    #             else:
    #                 if mat2048[i, j] > 0:
    #                     if st.pop() == mat2048[i, j]:
    #                         eq_row[i] += mat2048[i, j]
    #                     else:
    #                         st.append(mat2048[i, j])
    #                 else:
    #                     pass
    #                     # print(st)
    #     print(eq_row)
    #
    #     if imaxrow < 3 and mat2048[(imaxrow + 1):4, imaxcol].sum() == 0:
    #         move = 4  # 最大数不在最底下，同时最大数下面为空，下移
    #         print("最大数不在最底下，同时最大数下面为空，下移")
    #     elif imaxcol < 3 and mat2048[imaxrow, (imaxcol + 1):4].sum() == 0:
    #         move = 2  # 最大数不在最右边，同时最大数右边为空，右移
    #         print("最大数不在最右边，同时最大数右边为空，右移")
    #     elif eq_col.sum() >= eq_row.sum() and eq_col.sum() > 0:
    #         move = 4  # 如果向下合并可以合并更多,向下
    #         print("如果向下合并可以合并更多,向下")
    #     elif eq_row.sum() > eq_col.sum() and eq_row.sum() > 0:
    #         move = 2  # 如果向右可以合并更多,向右
    #         print("如果向右可以合并更多,向右")
    #     elif sum(mat2048[imaxrow, :]) > sum(mat2048[:, imaxcol]):
    #         # 向右，向下都没有合并的情况
    #         # 判断最大数那一行和列哪个数更多
    #         # 如果行更多，就向左，尽可能保证最大数不动
    #         move = random.choice((1, 2))
    #         print("向右，向下都没有，最大数所在行的数比较多，向左右随机")
    #     elif sum(mat2048[imaxrow, :]) < sum(mat2048[:, imaxcol]):
    #         # 向右，向下都没有合并的情况
    #         # 判断最大数那一行和列哪个数更多
    #         # 如果列更多，就向上，尽可能保证最大数不动
    #         move = random.choice((3, 4))
    #         print("向右，向下都没有，最大数所在列的数比较多，向上下随机")
    #     else:
    #         move = random.choice((1, 3))
    #         print("其他，随机右，下")
    #     return move

    def ai_train(self, epi=1):
        self.train = 1
        self.unbind_all("<Key>")
        for i in range(epi):
            # start = time.clock()
            self.playloops = 0
            # if i == 200:
            #     print("woshiyixia")
            #     pass
            self.score.reset_score()
            self.grid.reset_grid()
            for n in range(self.START_TILES):
                self.grid.pop_tile()
            # self.playloops = 0
            self.ai_pressed()
            # end = time.clock()
            # print(i, "时间：", end - start, 's', '循环次数：', self.playloops)
            print(i, '循环次数：', self.playloops)
        self.train = 0


if __name__ == "__main__":
    GabrieleCirulli2048().run()
# end if
