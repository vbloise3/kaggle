import sys
class Remover(object):
    def __init__(self, theList = [1,2,3,4,2,1,5,6,5,7,6,8,9,10,11,-1,-2,-1,-3,0]):
        self.theList = theList
    def remove_dups(self):
        _size = len(self.theList)
        self.theList.sort()
        duplicates = set()
        for item in range(_size -1, 0, -1):
            if self.theList[item] == self.theList[item - 1]:
                del self.theList[item]
        return self.theList

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_remover = Remover(input)
    else:
        my_remover = Remover()
    print(my_remover.remove_dups())