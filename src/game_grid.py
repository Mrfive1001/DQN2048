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
except:
    import tkinter as tk


# end try


def normalize(value, minimum=1):
    return max(abs(int(minimum)), abs(int(value)))


# end def


class GameGrid(tk.Canvas):
    """

    """
    BGCOLOR = "white"
    FGCOLOR = "grey"
    ROWS = 3
    COLUMNS = 3
    THICKNESS = 8  # pixels
    CONFIG = dict(
        background=BGCOLOR, highlightthickness=0, width=500, height=500
    )

    def __init__(self, master, **kw):
        tk.Canvas.__init__(self, master)
        self.CONFIG = self.CONFIG.copy()
        self.CONFIG.update(kw)
        self.configure(**self._only_tk(self.CONFIG))
        self.rows = kw.get("rows", self.ROWS)  # 行
        self.columns = kw.get("columns", self.COLUMNS)
        self.thickness = kw.get("thickness", self.THICKNESS)
        self.bgcolor = kw.get("bgcolor", self.BGCOLOR)
        self.fgcolor = kw.get("fgcolor", self.FGCOLOR)
        self.train = kw.get("train", 0)
        self.__tk_owner = master
        self.__tiles = dict()
        self.__matrix = GridMatrix(self.rows, self.columns)  # 几行几列的矩阵
        self.__cell_size = GridCellSize(self)
        self.init_widget(**self.CONFIG)


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

    @property
    def cell_size(self):
        return self.__cell_size

    # end def

    def clear_all(self, tk_event=None, *args, **kw):
        self.clear_grid()
        self.clear_tiles()
        self.matrix.reset_matrix()

    # end def

    def clear_grid(self, tk_event=None, *args, **kw):
        self.delete(tk.ALL)

    # end def

    def set_tiles(self, tiles):
        self.tiles.clear()
        self.__tiles = tiles

    def clear_tiles(self, tk_event=None, *args, **kw):
        self.tiles.clear()

    # end def

    @property
    def columns(self):
        return self.__columns

    # end def

    @columns.setter
    def columns(self, value):
        self.__columns = normalize(value)

    # end def

    @columns.deleter
    def columns(self):
        del self.__columns

    # end def

    def get_coords(self, row, column, centered=False):
        _x, _y = self.cell_size.xy_left_top(row, column)
        if centered:
            _x += self.cell_size.width // 2
            _y += self.cell_size.height // 2
        # end if
        return (_x, _y)

    # end def

    @property
    def grid_height(self):
        return self.winfo_reqheight()

    # end def

    @property
    def grid_size(self):
        return (
            (self.grid_width - self.half_high),
            (self.grid_height - self.half_high)
        )

    # end def

    @property
    def grid_width(self):
        return self.winfo_reqwidth()

    # end def

    @property
    def half_high(self):
        return round(0.1 + self.thickness / 2.0)

    # end def

    @property
    def half_low(self):
        return self.thickness // 2

    # end def

    def init_widget(self, **kw):
        pass

    # end def

    def is_full(self):
        return len(self.tiles) >= self.max_tiles

    # end def

    def is_tile(self, row, column):
        _x, _y = self.get_coords(row, column, centered=True)
        _item_id = self.find_overlapping(_x, _y, _x, _y)
        return bool(_item_id in self.tiles)

    # end def

    @property
    def matrix(self):
        return self.__matrix

    # end def

    @property
    def max_tiles(self):
        return self.rows * self.columns

    # end def

    @property
    def owner(self):
        return self.__tk_owner

    # end def

    def register_tile(self, tile_id, tile_object, raise_error=False):
        if tile_id not in self.tiles:
            self.tiles[tile_id] = tile_object
        elif raise_error:
            raise KeyError(
                "tile id '{tid}' is already registered."
                    .format(tid=tile_id)
            )
            # end if

    # end def

    def remove_tile(self, tile_id):
        self.tiles.pop(tile_id, None)

    # end def

    def reset_grid(self, tk_event=None, *args, **kw):
        self.clear_all()
        _grid_width, _grid_height = self.grid_size
        _x0, _y0 = self.xy_origin
        _thickness = self.thickness
        _fg = self.fgcolor
        self.create_rectangle(
            _x0, _y0, _grid_width, _grid_height,
            outline=_fg, width=_thickness,
        )
        for _column in range(1, self.columns):
            _x = _x0 + _column * (self.cell_size.width + _thickness)
            self.create_line(
                _x, 0, _x, _grid_height,
                fill=_fg, width=_thickness,
            )
        # end for
        for _row in range(1, self.rows):
            _y = _y0 + _row * (self.cell_size.height + _thickness)
            self.create_line(
                0, _y, _grid_width, _y,
                fill=_fg, width=_thickness,
            )
            # end for

    # end def

    @property
    def rows(self):
        return self.__rows

    # end def

    @rows.setter
    def rows(self, value):
        self.__rows = normalize(value)

    # end def

    @rows.deleter
    def rows(self):
        del self.__rows

    # end def

    @property
    def thickness(self):
        return self.__thickness

    # end def

    @thickness.setter
    def thickness(self, value):
        self.__thickness = normalize(value, minimum=0)

    # end def

    @thickness.deleter
    def thickness(self):
        del self.__thickness

    # end def

    @property
    def tiles(self):
        return self.__tiles

    # end def

    @property
    def xy_origin(self):
        _x0 = _y0 = self.half_low
        return (_x0, _y0)

    # end def

    @property
    def xy_center(self):
        return (self.grid_width // 2, self.grid_height // 2)
        # end def


# end class


class GridAnimation(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.owner = master
        self.__pid = 0
        self.__animation_kw = dict()
        self.__callback = None
        self.__callback_args = tuple()
        self.__callback_kw = dict()

    # end def

    @property
    def keywords(self):
        return self.__animation_kw

    # end def

    def register(self, callback, *args, **kw):
        if callable(callback):
            self.__callback = callback
            self.__callback_args = args
            self.__callback_kw = kw
            return True
        else:
            raise TypeError(
                "callback object *MUST* be a callable one."
            )
        # end if - callable
        return False

    # end def

    def resume(self):
        return self.run_sequencer()

    # end def

    def run_sequencer(self, animation_kw=None):
        self.stop()
        if callable(self.__callback):
            if isinstance(animation_kw, dict):
                self.__animation_kw = _anim_kw = animation_kw
            else:
                _anim_kw = self.__animation_kw
            # end if - animation_kw
            _sequence = _anim_kw.get("sequence")
            if isinstance(_sequence, (list, tuple)):
                _interval = int(_anim_kw.get("interval", 100))
                _step = int(_anim_kw.get("step", 0))
                if _step < len(_sequence):
                    self.__callback_kw.update(
                        value=_sequence[_step]
                    )
                    self.__callback(
                        *self.__callback_args, **self.__callback_kw
                    )
                    self.__animation_kw["step"] = _step + 1
                    self.__pid = self.after(
                        _interval, self.run_sequencer
                    )
                    # end if - new step
                    # end if - sequence
        # end if - callable
        return self.__pid

    # end def

    def start(self, interval=100, step=0, sequence=None):
        return self.run_sequencer(
            dict(interval=interval, step=step, sequence=sequence)
        )

    # end def

    def start_after(self, delay=500, interval=100, step=0, sequence=None):
        self.__pid = self.after(
            delay, self.start, interval, step, sequence
        )
        return self.__pid

    # end def

    def stop(self, pid=None):
        if pid:
            self.after_cancel(pid)
        else:
            self.after_cancel(self.__pid)
            self.__pid = 0
            # end if - pid
            # end def


# end class


class GridCellSize:
    def __init__(self, grid_owner):
        self.__tk_owner = grid_owner
        self.__width = None
        self.__height = None

    # end def

    def _real_size(self, size, count, thickness):
        _size = size - (count + 1) * thickness
        return round(abs(_size // count))

    # end def

    @property
    def height(self):
        if not self.__height:
            self.__height = self._real_size(
                size=self.owner.grid_height,
                count=self.owner.rows,
                thickness=self.owner.thickness,
            )
        # end if
        return self.__height

    # end def

    @property
    def owner(self):
        return self.__tk_owner

    # end def

    @property
    def size(self):
        return (self.width, self.height)

    # end def

    @property
    def size_hxw(self):
        return (self.height, self.width)

    # end def

    @property
    def size_wxh(self):
        return (self.width, self.height)

    # end def

    @property
    def width(self):
        if not self.__width:
            self.__width = self._real_size(
                size=self.owner.grid_width,
                count=self.owner.columns,
                thickness=self.owner.thickness,
            )
        # end if
        return self.__width

    # end def

    def x_center(self, column):
        return self.x_left(column) + self.width // 2

    # end def

    def x_left(self, column):
        _column = min(abs(int(column)), self.owner.columns)
        _thickness = self.owner.thickness
        _x = _thickness + _column * (self.width + _thickness)
        return _x

    # end def

    def xy_center(self, row, column):
        return (self.x_center(column), self.y_center(row))

    # end def

    def xy_left_top(self, row, column):
        return (self.x_left(column), self.y_top(row))

    # end def

    def y_center(self, row):
        return self.y_top(row) + self.height // 2

    # end def

    def y_top(self, row):
        _row = min(abs(int(row)), self.owner.rows)
        _thickness = self.owner.thickness
        _y = _thickness + _row * (self.height + _thickness)
        return _y
        # end def


# end class


class GridError(Exception):
    pass


# end class


class GridMatrix:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        # matrix is a dict, the key means row, value is column
        # but what use for? tile also have the row and column value
        self.reset_matrix()

    # end def

    def add(self, object_, row, column, raise_error=False):
        # 将某一个值置为object_
        if self.matrix.get((row, column)) is None:
            self.matrix[(row, column)] = object_
            return True
        elif raise_error:
            raise GridError(
                "cannot add object at (row, column) = "
                "({row}, {col}): busy location."
                    .format(row=row, col=column)
            )
        # end if
        return False

    # end def

    @property
    def columns(self):
        return self.__columns

    # end def

    @columns.setter
    def columns(self, value):
        self.__columns = normalize(value)

    # end def

    @columns.deleter
    def columns(self):
        del self.__columns

    # end def

    def duplicate_object(self, from_row_column, to_row_column):
        _object = self.get_object_at(*from_row_column, raise_error=True)
        self.add(_object, *to_row_column, raise_error=True)

    # end def

    def get_object_at(self, row, column, raise_error=False):
        # 从矩阵中得到某个位置的量
        _object = self.matrix.get((row, column))
        if raise_error and _object is None:
            raise GridError(
                "no object found at (row, column) = "
                "({row}, {col}): empty location."
                    .format(row=row, col=column)
            )
        # end if
        return _object

    # end def

    @property
    def matrix(self):
        return self.__matrix

    # end def

    def move_object(self, from_row_column, to_row_column):
        # 从一个位置移动到另外一个覆盖掉
        _object = self.get_object_at(*from_row_column, raise_error=True)
        self.add(_object, *to_row_column, raise_error=True)
        self.remove_object_at(*from_row_column)

    # end def

    def remove_object_at(self, row, column):
        # 删除某个位置的值
        self.matrix.pop((row, column), None)

    # end def

    def reset_matrix(self):
        # 重置
        self.__matrix = dict()

    # end def

    @property
    def rows(self):
        return self.__rows

    # end def

    @rows.setter
    def rows(self, value):
        self.__rows = normalize(value)

    # end def

    @rows.deleter
    def rows(self):
        del self.__rows

    # end def

    def swap_objects(self, row_column1, row_column2):
        _object1 = self.get_object_at(*row_column1, raise_error=True)
        _object2 = self.get_object_at(*row_column2, raise_error=True)
        self.remove_object_at(*row_column1)
        self.remove_object_at(*row_column2)
        self.add(_object1, *row_column2, raise_error=True)
        self.add(_object2, *row_column1, raise_error=True)
        # end def


# end class


class GridTile:
    def __init__(self, grid_owner, value, row, column, train = 0):
        self.__tk_owner = grid_owner
        self.__cell_size = grid_owner.cell_size
        self.tag = "GridTile{}".format(id(self))
        self.id = None
        self.value = value
        self.row = row
        self.column = column
        self.train = train

    # end def

    @property
    def cell_size(self):
        return self.__cell_size

    # end def

    @property
    def column(self):
        return self.__column

    # end def

    @column.setter
    def column(self, value):
        self.__column = normalize(value, minimum=0)

    # end def

    @column.deleter
    def column(self):
        del self.__column

    # end def

    @property
    def row_column(self):
        return (self.row, self.column)

    # end def

    @property
    def owner(self):
        return self.__tk_owner

    # end def

    @property
    def row(self):
        return self.__row

    # end def

    @row.setter
    def row(self, value):
        self.__row = normalize(value, minimum=0)

    # end def

    @row.deleter
    def row(self):
        del self.__row

    # end def

    @property
    def size(self):
        return self.cell_size.size_wxh

    # end def

    @property
    def value(self):
        return self.__value

    # end def

    @value.setter
    def value(self, new_value):
        self.__value = new_value

    # end def

    @value.deleter
    def value(self):
        del self.__value

    # end def

    @property
    def xy_center(self):
        return self.cell_size.xy_center(self.row, self.column)

    # end def

    @property
    def xy_origin(self):
        return self.cell_size.xy_left_top(self.row, self.column)
        # end def

# end class
