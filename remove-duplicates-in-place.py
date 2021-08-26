# How do you remove duplicates from an array in place
import sys
class Duplicates(object):
    def __init__(self, theList = [1,4,2,7,6,4,7,8,9,20]):
        self.theList = theList
    def remove_in_place(self):
        self.theList = sorted(self.theList)
        for i in range(len(self.theList) - 1, 0, -1):
            if self.theList[i] == self.theList[i - 1]:
                del self.theList[i]
        return self.theList

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        my_dups = Duplicates(input)
    else:
        my_dups = Duplicates()
    print(my_dups.remove_in_place())
