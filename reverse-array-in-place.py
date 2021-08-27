# How do you reverse an array in place
import sys
class Reverser(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9]):
        self.theList = theList
    def reverse_in_place(self):
        _size = len(self.theList)
        i = 0
        j = _size - 1
        while i < j:
            self.swap(i, j)
            i += 1
            j -= 1
        return self.theList
    def swap(self, i, j):
        temp = self.theList[i]
        self.theList[i] = self.theList[j]
        self.theList[j] = temp

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        my_reverser = Reverser(input)
    else:
        my_reverser = Reverser()
    print(my_reverser.reverse_in_place())