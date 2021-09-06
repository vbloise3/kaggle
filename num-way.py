import sys
class Finder(object):
    def __init__(self, theSteps = 4):
        self.theSteps = theSteps
    def find_ways(self, num):
        if num == 0 or num == 1:
            return 1
        else:
            return self.find_ways(num - 1) + self.find_ways(num - 2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
        my_finder = Finder(steps)
    else:
        my_finder = Finder()
    for i in range(int(sys.argv[1]) + 1):
        print(my_finder.find_ways(i))