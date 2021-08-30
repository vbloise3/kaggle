import sys
class Finder(object):
    def __init__(self, theList = [-25,1,2,4,5,6,7,8,9]):
        self.theList = theList
    def find_missing_number(self):
        _size = len(self.theList)
        found = set()
        self.theList.sort()
        for item in self.theList:
            found.add(item)
        for j in range(min(self.theList), max(self.theList)):
            if j not in found:
                return j
        return -1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv
        input = list(map(int, arguments[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.find_missing_number())