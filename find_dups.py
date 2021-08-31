import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,6,1,8,7,3,9,6,7,9,10]):
        self.theList = theList
    def find_dups(self):
        duplicates = set()
        non_duplicates = set()
        for item in self.theList:
            if item in non_duplicates:
                duplicates.add(item)
            else:
                non_duplicates.add(item)
        return(list(duplicates))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.find_dups())