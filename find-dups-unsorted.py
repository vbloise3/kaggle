import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,2,4,6,7,8,9,-1,-3,-1,-2,-4,-2,-3]):
        self.theList = theList
    def find_dups(self):
        non_dups = set()
        dups = set()
        for item in self.theList:
            if item not in non_dups:
                non_dups.add(item)
            else:
                dups.add(item)
        return dups

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv.split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.find_dups())