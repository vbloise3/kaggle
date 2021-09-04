import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,2,6,4,8,5,6,78,1,3,6,78]):
        self.theList = theList
    def remove_dups(self):
        self.theList.sort()
        duplicates = set()
        _size = len(self.theList)
        for j in range(_size - 1, 0, -1):
            if self.theList[j] == self.theList[j - 1]:
                del self.theList[j]
                #duplicates.add(self.theList[j])
        return self.theList #list(duplicates)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print (my_finder.remove_dups())