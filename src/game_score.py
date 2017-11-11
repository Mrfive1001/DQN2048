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

try:
    import Tkinter as tk
    import ttk
except:
    import tkinter as tk
    from tkinter import ttk
# end try


class GameScore (ttk.Frame):

    CONFIG = dict(padding=5)

    def __init__(self, master=None, **kw):
        ttk.Frame.__init__(self, master)
        self.CONFIG = self.CONFIG.copy()
        self.CONFIG.update(kw)
        self.configure(**self._only_tk(self.CONFIG))
        self._cvar = tk.IntVar()
        self.init_widget(**self.CONFIG)
    # end def

    def _bind_high(self, value):
        return value
    # end def

    def _bind_low(self, value):
        return max(0, value)
    # end def

    def _only_tk(self, kw):
        _dict = dict()
        if hasattr(self, "tk") and hasattr(self, "configure"):
            _attrs = set(self.configure().keys()) & set(kw.keys())
            for _key in _attrs:
                _dict[_key] = kw.get(_key)
            # end for
        # end if
        return _dict
    # end def

    def add_score(self, value):
        self._cvar.set(
            self._bind_high(
                self._cvar.get() + abs(int(value))
            )
        )
    # end def

    def get_score(self):
        return self._cvar.get()
    # end def

    def high_score(self, value):
        self._cvar.set(max(self._cvar.get(), int(value)))
    # end def

    def init_widget(self, **kw):
        self.reset_score()
        self.score_label = ttk.Label(
            self, text=kw.get("label", "Score:"),
        )
        self.score_label.pack(side=tk.LEFT)
        self.score_display = ttk.Label(
            self, textvariable=self._cvar,
        )
        self.score_display.pack(side=tk.RIGHT)
    # end def

    def reset_score(self):
        self._cvar.set(0)
    # end def

    def set_score(self, value):
        self._cvar.set(int(value))
    # end def

    def sub_score(self, value):
        self._cvar.set(
            self._bind_low(
                self._cvar.get() - abs(int(value))
            )
        )
    # end def

# end class
