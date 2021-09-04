import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9,-1,-2,3,-5,3,2,6]):
        self.theList = theList
    def get_sequence(self):
        _size = len(self.theList)
        max_so_far = -sys.maxsize -1
        max_ending_here = 0
        for item in range(_size):
            max_ending_here = max_ending_here + self.theList[item]
            if max_so_far < max_ending_here:
                max_so_far = max_ending_here
        return max_so_far

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.get_sequence())