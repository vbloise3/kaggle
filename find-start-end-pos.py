import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,3,3,3,4,5,6,7,8,9], theValue = 3):
        self.theList = theList
        self.theValue = theValue
    def get_sequence(self):
        _size = len(self.theList)
        first = -1
        last = -1
        for item in range(_size):
            if self.theValue != self.theList[item]:
                continue
            if first == -1:
                first = item
            last = item
        if first != -1:
            return(str(first), str(last))

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input = list(map(int, sys.argv[1].split(",")))
        value = int(sys.argv[2])
        my_finder = Finder(input, value)
    else:
        my_finder = Finder()
    print(my_finder.get_sequence())