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

try:
    import Tkinter as tk
    import ttk
except:
    import tkinter as tk
    from tkinter import ttk
# end try

from . import game_grid as GG


class Game2048Grid (GG.GameGrid):

    BGCOLOR = "#ccc0b3"
    FGCOLOR = "#bbada0"
    ROWS = COLUMNS = 4
    THICKNESS = 8   # pixels
    CONFIG = dict(
        background=BGCOLOR, highlightthickness=0, width=400, height=400
    )

    def animate_rectangle(self, item_id, value):
        self.tag_raise(item_id, tk.ALL)
        self.itemconfigure(item_id, stipple=value)
    # end def

    def animate_text_game_over(self, item_id, value):
        self.tag_raise(item_id, tk.ALL)
        self.itemconfigure(item_id, fill=value, state=tk.NORMAL)
    # end def

    def animate_text_try_again(self, item_id, value):
        """
        self.tag_raise(item_id, tk.ALL)
        self.itemconfigure(item_id, fill=value, state=tk.NORMAL)
        if value == "#ffffff":
            _btn = ttk.Button(
                self,
                text="Play",
                command=self.owner.new_game
            )
            self.create_window(
                self.winfo_reqwidth() // 2,
                self.winfo_reqheight() // 2 + 65,
                window=_btn,
            )
        """
        # end if
    # end def

    def fuse_tiles(self, into_tile, void_tile):
        _into, _void = into_tile, void_tile
        if _into and _void and (_into.value == _void.value):
            _into.value += _void.value
            self.update_score(_into.value)
            _into.update_display()
            self.matrix.remove_object_at(*_void.row_column)
            self.remove_tile(_void.id)
            _void.animate_remove()
            return True
        # end if - matching tiles
        return False
    # end def

    def game_over(self, tk_event=None, *args, **kw):
        self.unbind_all("<Key>")
        _grid_width = self.winfo_reqwidth()
        _grid_height = self.winfo_reqheight()
        _rect_id = self.create_rectangle(
            0, 0, _grid_width, _grid_height,
            fill=self.FGCOLOR, width=0,
        )
        _anim_rect = GG.GridAnimation(self)
        _anim_rect.register(
            self.animate_rectangle, item_id=_rect_id,
        )
        _anim_rect.start(sequence=("gray12", "gray25", "gray50"))
        _text_id = self.create_text(
            _grid_width // 2, _grid_height // 2 - 25,
            text="GAME OVER", font="sans 32 bold", fill="white",
            state=tk.HIDDEN,
        )
        _anim_text1 = GG.GridAnimation(self)
        _anim_text1.register(
            self.animate_text_game_over, item_id=_text_id,
        )
        _anim_text1.start_after(
            delay=800, interval=50,
            sequence=("#c9bdb4", "#d0c5be", "#d7cdc8", "#ded5d2",
                      "#e5dddc", "#ece5e6", "#f3edf0", "#ffffff"),
        )
        _text_id = self.create_text(
            _grid_width // 2, _grid_height // 2 + 30,
            text="Try again", font="sans 16 bold", fill="white",
            state=tk.HIDDEN,
        )
        _anim_text2 = GG.GridAnimation(self)

        _anim_text2.register(
            self.animate_text_try_again, item_id=_text_id,
        )
        _anim_text2.start_after(
            delay=1600, interval=80,
            sequence=("#c9bdb4", "#d0c5be", "#d7cdc8", "#ded5d2",
                      "#e5dddc", "#ece5e6", "#f3edf0", "#ffffff"),
        )
    # end def

    def get_available_box(self):
        if self.is_full():
            raise GG.GridError("no more room in grid")
        else:
            _at = self.matrix.get_object_at
            while True:
                _row = random.randrange(self.rows)
                _column = random.randrange(self.columns)
                if not _at(_row, _column):
                    break
                # end if
            # end while
            return (_row, _column)
        # end if - no more room
    # end def

    def init_widget(self, **kw):
        self.__score_cvar = tk.IntVar()
        self.__score_callback = None
    # end def

    def move_tile(self, tile, row, column):
        if tile:
            self.matrix.move_object(tile.row_column, (row, column))
            tile.animate_move_to(row, column)
        # end if
    # end def

    def move_tiles_down(self):
        _at = self.matrix.get_object_at
        _acted = False
        for _column in range(self.columns):
            for _row in range(self.rows - 1, -1, -1):
                _tile1 = _at(_row, _column)
                if _tile1:
                    for _row2 in range(_row - 1, -1, -1):
                        _tile2 = _at(_row2, _column)
                        if self.fuse_tiles(_tile1, _tile2):
                            _acted = True
                        # end if
                        if _tile2:
                            break
                    # end for - next tile
                # end if - tile
            # end for - fusions
            _empty = None
            for _row in range(self.rows - 1, -1, -1):
                _tile1 = _at(_row, _column)
                if not _tile1 and not _empty:
                    _empty = (_row, _column)
                elif _tile1 and _empty:
                    self.move_tile(_tile1, *_empty)
                    _empty = (_empty[0] - 1, _column)
                    _acted = True
                # end if
            # end for - scrollings
        # end for - row
        self.next_tile(acted=_acted)
    # end def

    def move_tiles_left(self):
        _at = self.matrix.get_object_at
        _acted = False
        for _row in range(self.rows):
            for _column in range(self.columns - 1):
                _tile1 = _at(_row, _column)
                if _tile1:
                    for _col in range(_column + 1, self.columns):
                        _tile2 = _at(_row, _col)
                        if self.fuse_tiles(_tile1, _tile2):
                            _acted = True
                        # end if
                        if _tile2:
                            break
                    # end for - next tile
                # end if - tile
            # end for - fusions
            _empty = None
            for _column in range(self.columns):
                _tile1 = _at(_row, _column)
                if not _tile1 and not _empty:
                    _empty = (_row, _column)
                elif _tile1 and _empty:
                    self.move_tile(_tile1, *_empty)
                    _empty = (_row, _empty[1] + 1)
                    _acted = True
                # end if
            # end for - scrollings
        # end for - row
        self.next_tile(acted=_acted)
    # end def

    def move_tiles_right(self):
        _at = self.matrix.get_object_at
        _acted = False
        for _row in range(self.rows):
            for _column in range(self.columns - 1, -1, -1):
                _tile1 = _at(_row, _column)
                if _tile1:
                    for _col in range(_column - 1, -1, -1):
                        _tile2 = _at(_row, _col)
                        if self.fuse_tiles(_tile1, _tile2):
                            _acted = True
                        # end if
                        if _tile2:
                            break
                    # end for - next tile
                # end if - tile
            # end for - fusions
            _empty = None
            for _column in range(self.columns - 1, -1, -1):
                _tile1 = _at(_row, _column)
                if not _tile1 and not _empty:
                    _empty = (_row, _column)
                elif _tile1 and _empty:
                    self.move_tile(_tile1, *_empty)
                    _empty = (_row, _empty[1] - 1)
                    _acted = True
                # end if
            # end for - scrollings
        # end for - row
        self.next_tile(acted=_acted)
    # end def

    def move_tiles_up(self):
        _at = self.matrix.get_object_at
        _acted = False
        for _column in range(self.columns):
            for _row in range(self.rows - 1):
                _tile1 = _at(_row, _column)
                if _tile1:
                    for _row2 in range(_row + 1, self.rows):
                        _tile2 = _at(_row2, _column)
                        if self.fuse_tiles(_tile1, _tile2):
                            _acted = True
                        # end if
                        if _tile2:
                            break
                    # end for - next tile
                # end if - tile
            # end for - fusions
            _empty = None
            for _row in range(self.rows):
                _tile1 = _at(_row, _column)
                if not _tile1 and not _empty:
                    _empty = (_row, _column)
                elif _tile1 and _empty:
                    self.move_tile(_tile1, *_empty)
                    _empty = (_empty[0] + 1, _column)
                    _acted = True
                # end if
            # end for - scrollings
        # end for - row
        self.next_tile(acted=_acted)
    # end def

    def next_tile(self, tk_event=None, *args, **kw):
        if kw.get("acted"):
            self.pop_tile()
        # end if
        if self.no_more_hints():
            self.game_over()
        # end if
    # end def

    def no_more_hints(self):
        if self.is_full():
            _at = self.matrix.get_object_at
            for _row in range(self.rows):
                for _column in range(self.columns):
                    _tile1 = _at(_row, _column)
                    _tile2 = _at(_row, _column + 1)
                    _tile3 = _at(_row + 1, _column)
                    if _tile1 and (
                        (_tile2 and _tile1.value == _tile2.value) or
                            (_tile3 and _tile1.value == _tile3.value)):
                        return False
                    # end if
                # end for - columns
            # end for - rows
            return True
        # end if - no more room
        return False
    # end def

    def pop_tile(self, tk_event=None, *args, **kw):
        if not self.is_full():
            _value = random.choice([2, 4, 2, 2])
            _row, _column = self.get_available_box()
            _tile = Game2048GridTile(self, _value, _row, _column)
            _tile.animate_show()
            self.register_tile(_tile.id, _tile)
            self.matrix.add(_tile, *_tile.row_column, raise_error=True)
        # end if - room in grid
    # end def

    def set_score_callback(self, callback, raise_error=False):
        if callable(callback):
            self.__score_callback = callback
        elif raise_error:
            raise TypeError(
                "callback parameter *MUST* be a callable object."
            )
        # end if
    # end def

    def tiles_match(self, tile1, tile2):
        return tile1 and tile2 and tile1.value == tile2.value
    # end def

    def update_score(self, value, mode="add"):
        if callable(self.__score_callback):
            self.__score_callback(value, mode)
        # end if
    # end def

# end class


class Game2048GridTile (GG.GridTile):
    COLORS = {
        2: ("#eee4da", "#776e65"),
        4: ("#ede0c8", "#776e65"),
        8: ("#f2b179", "#f9f6f2"),
        16: ("#f59563", "#f9f6f2"),
        32: ("#f67c5f", "#f9f6f2"),
        64: ("#f65e3b", "#f9f6f2"),
        128: ("#edcf72", "#f9f6f2"),
        256: ("#edcc61", "#f9f6f2"),
        512: ("#edc850", "#f9f6f2"),
        1024: ("#edc53f", "#f9f6f2"),
        2048: ("#edc22e", "#f9f6f2"),
        4096: ("#ed952e", "#ffe0b7"),
        8192: ("#d2ff50", "#bb6790"),
        16384: ("yellow", "chocolate"),
        32768: ("orange", "yellow"),
        65536: ("red", "white"),
    }
    FONTS = {
        2: "sans 32 bold",
        4: "sans 32 bold",
        8: "sans 32 bold",
        16: "sans 28 bold",
        32: "sans 28 bold",
        64: "sans 28 bold",
        128: "sans 24 bold",
        256: "sans 24 bold",
        512: "sans 24 bold",
        1024: "sans 20 bold",
        2048: "sans 20 bold",
        4096: "sans 20 bold",
        8192: "sans 20 bold",
        16384: "sans 16 bold",
        32768: "sans 16 bold",
        65536: "sans 16 bold",
    }

    def animate_move_to(self, row, column):
        _x0, _y0 = self.xy_origin
        _x1, _y1 = self.cell_size.xy_left_top(row, column)
        self.owner.move(self.tag, (_x1 - _x0), (_y1 - _y0))
        self.row, self.column = row, column
    # end def

    def animate_tile_popup(self, value):
        _x0, _y0 = self.xy_center
        self.owner.scale(self.id, _x0, _y0, value, value)
    # end def

    def animate_remove(self):
        self.owner.delete(self.tag)
    # end def

    def animate_show(self):
        _x, _y = self.xy_origin
        _width, _height = self.size
        _bg, _fg = self.get_value_colors()
        self.id = self.owner.create_rectangle(
            _x, _y, (_x + _width), (_y + _height),
            fill=_bg, width=0, tags=(self.tag, "tiles"),
        )
        _font = self.get_value_font()
        _x, _y = self.xy_center
        self.value_id = self.owner.create_text(
            _x, _y, text=str(self.value),
            fill=_fg, font=_font, tags=(self.tag, "values"),
        )
        _anim_tile = GG.GridAnimation()
        _anim_tile.register(self.animate_tile_popup)
        _anim_tile.start(
            interval=50, sequence=(6.0 / 5.0, 6.0 / 5.0, 5.0 / 6.0, 5.0 / 6.0),
        )
    # end def

    def get_value_colors(self):
        return self.COLORS.get(self.value, ("red", "yellow"))
    # end def

    def get_value_font(self):
        return self.FONTS.get(self.value, "sans 10 bold")
    # end def

    def update_display(self, tk_event=None, *args, **kw):
        _bg, _fg = self.get_value_colors()
        self.owner.itemconfigure(self.id, fill=_bg)
        self.owner.itemconfigure(
            self.value_id,
            text=str(self.value),
            font=self.get_value_font(),
            fill=_fg,
        )
    # end def

# end class
