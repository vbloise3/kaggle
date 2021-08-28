# Given an array of integers sorted in ascending order, find the starting and ending position of a given value
import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,3,3,4,5,6,6,6,6,7], value = 6):
        self.theList = theList
        self.theValue = value
    def find(self):
        _size = len(self.theList)
        first = -1
        last = -1
        for i in range(_size):
            if self.theValue != self.theList[i]:
                continue
            if first == -1:
                first = i
            last = i
        if first != -1:
            return (str(first), str(last))

if __name__ == "__main__":
    if len(sys.argv) > 2:
        arguments = sys.argv
        input_list = list(map(int, arguments[1].split(",")))
        input_value = int(arguments[2])
        my_finder = Finder(input_list, input_value)
    else:
        my_finder = Finder()
    print(my_finder.find())