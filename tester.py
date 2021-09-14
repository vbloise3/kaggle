import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9,0]):
        self.theList = theList
    def test(self):
        _size = len(self.theList)
        for i in range(_size - 1, -1, -1):
            print(self.theList[i])
        return _size

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.test())