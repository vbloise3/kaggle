import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,-1,4,5,6,-1,3,-5,2]):
        self.theList = theList
    def find_longest_sequence(self):
        _size = len(self.theList)
        longest_so_far = 1
        max_longest = 0
        for i in range(_size - 1):
            if self.theList[i] + 1 == self.theList[i + 1]:
                longest_so_far += 1
                max_longest = longest_so_far
            else:
                longest_so_far = 1
        return max(longest_so_far, max_longest)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.find_longest_sequence())