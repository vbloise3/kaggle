import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,6,7,8,9,10]):
        self.theList = theList
    def find_missing_number(self):
        for i in range(len(self.theList)):
            if self.theList[i] + 1 != self.theList[i + 1]:
                return self.theList[i] + 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.find_missing_number())
