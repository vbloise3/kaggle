# Given an integer array, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum
import sys
class Finder(object):
    def __init__(self, theList = [-2,-3,4,-1,-2,1,5,-3]):
        self.theList = theList
    def largest_contiguous_sum(self):
        _size = len(self.theList)
        max_so_far = -sys.maxsize - 1
        max_ending_here = 0
        for i in range(_size):
            max_ending_here = max_ending_here + self.theList[i]
            if (max_so_far < max_ending_here):
                max_so_far = max_ending_here
    
            if max_ending_here < 0:
                max_ending_here = 0  
        return max_so_far

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv
        input = list(map(int, arguments[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.largest_contiguous_sum())