# Rotate an array left and right by a given number k
import sys
class Rotater(object):
    def __init__(self, theList = [1,2,5,67,8,9,43], value = 3):
        self.theListL = theList.copy()
        self.theListR = theList.copy()
        self.theValue = value
        self._size = len(self.theListL)
    def rotate_left(self):
        for i in range(self.theValue):
            temp = self.theListL[0]
            for j in range(self._size - 1):
                self.theListL[j] = self.theListL[j + 1]
            self.theListL[self._size - 1] = temp
        return self.theListL
    def rotate_right(self):
        for i in range(self.theValue):
            temp = self.theListR[self._size - 1]
            for j in range(self._size - 1, 0, -1):
                self.theListR[j] = self.theListR[j - 1]
            self.theListR[0] = temp
        return self.theListR
if __name__ == "__main__":
    if len(sys.argv) > 2:
        arguments = sys.argv
        input_list = list(map(int, arguments[1].split(",")))
        input_value = int(arguments[2])
        my_rotater = Rotater(input_list, input_value)
    else:
        my_rotater = Rotater()
    print("Rotated left: ", my_rotater.rotate_left())
    print("Rotated right: ", my_rotater.rotate_right())