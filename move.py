import itertools

Size = 4


class UpdateNew(object):
    """docstring for UpdateNew"""

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
                # if matrix.min() == 0 and (matrix!=lastmatrix).any():       #矩阵中有最小值0且移动后的矩阵不同，才可以添加0位置处添加随机数
                # GameInit.initData(Size,matrix,self.zerolist)
        return matrix


class LeftAction(UpdateNew):
    """docstring for LeftAction"""

    def __init__(self, matrix):
        super(LeftAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy()  # 获得一份矩阵的复制
        newmatrix = self.toSequence(matrix)
        return newmatrix


class RightAction(UpdateNew):
    """docstring for RightAction"""

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
    """docstring for DownAction"""

    def __init__(self, matrix):
        super(DownAction, self).__init__(matrix)

    def handleData(self):
        matrix = self.matrix.copy()[::-1].T
        newmatrix = self.toSequence(matrix)
        return newmatrix.T[::-1]


class TestScore(UpdateNew):
    # 检测矩阵中的零元素
    def __init__(self, matrix):
        super(TestScore, self).__init__(matrix)

    def EmptyTest(self):
        mat_list = self.matrix.tolist()
        # print(mat_list)
        out = list(itertools.chain.from_iterable(mat_list))
        score = out.count(0.0)
        return score * 110

    # jiance hang lie de dan diao xing
    def Monotonicity(self):
        mat_mono = self.matrix.copy()
        score1 = 0
        score2 = 0
        row, colu = mat_mono.shape
        for j in range(colu):
            for i in range(row - 1):
                if mat_mono[i + 1, j] >= mat_mono[i, j]:
                    score1 += 1 * i
        for i in range(row):
            for j in range(colu - 1):
                if mat_mono[i, j + 1] <= mat_mono[i, j]:
                    score2 += 1 * (4 - j)

        return (score1 + score2) * 5

    def ALLnum(self):
        mat = self.matrix.copy()
        score = 0
        row, colu = mat.shape
        for j in range(colu):
            for i in range(row):
                score += mat[i, j]
        return score / 2

    # 检测同行是否有相同的元素
    def equall(self):
        mat = self.matrix.copy()
        score = 0
        row, colu = mat.shape
        for i in range(row):
            for j in range(colu - 1):
                if mat[i, j] == mat[i, j + 1]:
                    score += mat[i, j]
        return score * 5

    # 检测最大值是否在左下角
    def wheremax(self):
        mat = self.matrix.copy()
        score = 0
        imax = mat.argmax()
        imaxrow = int(imax / 4)
        imaxcol = imax - imaxrow * 4
        if imaxrow == 3 and imaxcol == 0:
            score += 180
            if mat[2, 0] >= 256:
                score += 120
                if mat[1, 0] >= 128:
                    score += 120
                    if mat[0, 0] >= 64:
                        score += 120
                        if mat[0, 1] >= 64:
                            score += 120
        return score * 1.5
