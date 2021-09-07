import sys
class Reverser(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9,0]):
        self.theList = theList
    def reverse(self):
        i = 0
        j = len(self.theList) - 1
        while i < j:
            self.swap(i, j)
            i += 1
            j -= 1
        return self.theList
    def swap(self, i, j):
        temp = self.theList[i]
        self.theList[i] = self.theList[j]
        self.theList[j] = temp

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_reverser = Reverser(input)
    else:
        my_reverser = Reverser()
    print(my_reverser.reverse())