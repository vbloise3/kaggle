import sys
class Rotater(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9,0], theValue = 4):
        self.theListL = theList.copy()
        self.theValue = theValue
        self.theListR = theList.copy()
        self._size = len(theList)
    def rotateLeft(self):
        for i in range(self.theValue):
            temp = self.theListL[0]
            for j in range(self._size - 1):
                self.theListL[j] = self.theListL[j + 1]
            self.theListL[self._size - 1] = temp
        return self.theListL
    def rotateRight(self):
        for i in range(self.theValue):
            temp = self.theListR[self._size - 1]
            for j in range(self._size - 1, 0, -1):
                self.theListR[j] = self.theListR[j - 1]
            self.theListR[0] = temp
        return self.theListR

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input = list(map(int, sys.argv[1].split(",")))
        value = int(sys.argv[2])
        my_rotater = Rotater(input, value)
    else:
        my_rotater = Rotater()
    print(my_rotater.rotateLeft())
    print(my_rotater.rotateRight())