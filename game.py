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
import matplotlib.pyplot as plt

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
import move


class GabrieleCirulli2048(tk.Tk):
    PADDING = 10  # 控制界面的
    START_TILES = 2  # 初始化几个方格

    def __init__(self, **kw):
        tk.Tk.__init__(self)

        self.train = kw.get("train", 0)  # 从类读取是否训练
        self.rule = kw.get("rule", 1)  # 从类读取是否训练
        self.ai_time = kw.get('ai_time', 12)
        self.initialize(**kw)  # 画图的初始化

    def run(self, **kw):
        # 主程序的执行过程，如果训练那就不画图了
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
        if self.train == 0:
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
        self.after(self.ai_time, self.step)  # 多长时间后调用下一次
        self.bind_all("<Key>", self.on_keypressed)

    # 规则式选择动作
    def ai_rule(self, mate):
        mat = mate.copy()  # 对数形式
        next_ = [move.LeftAction(mat).handleData(), move.RightAction(mat).handleData(),
                 move.UpAction(mat).handleData(), move.DownAction(mat).handleData()]
        sco = []
        for st in next_:
            if (st == mat).all():
                sco.append(-10)
            else:
                st_ = move.TestScore(st)
                sco_ = st_.evaluate()
                sco.append(sco_)
        pp = np.array(sco).argmax()
        return pp

    def step(self):
        # 可以返回状态、动作和奖励
        mat2048_old = np.zeros((4, 4))
        tiles = self.grid.tiles
        for t in tiles:
            mat2048_old[tiles[t].row, tiles[t].column] = tiles[t].value
        # 2048表格的2对数
        pressed = self.ai_rule(mat2048_old)
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
        done = self.grid.no_more_hints()
        if done:
            pass
        else:
            self.after(self.ai_time, self.step)


if __name__ == "__main__":
    ai = GabrieleCirulli2048()
    ai.run()
