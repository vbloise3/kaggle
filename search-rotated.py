import sys
class Finder(object):
    def __init__(self, theList = [5,6,7,8,9,10,11,0,1,2,3,4], value = 4):
        self.theList = theList
        self.value = value
    def search_rotated(self):
        low = 0
        high = len(self.theList)
        while low < high:
            mid = low + (high - low) // 2
            if self.value == self.theList[mid]:
                return mid
            if self.theList[low] <= self.theList[mid]:
                if self.value >= self.theList[low] and self.value < self.theList[mid]:
                    high = mid
                else:
                    low = mid + 1
            else:
                if self.value <= self.theList[high - 1] and self.value > self.theList[mid]:
                    low = mid + 1
                else:
                    high = mid
        return -1

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input = list(map(int, sys.argv[1].split(",")))
        value = int(sys.argv[2])
        my_finder = Finder(input, value)
    else:
        my_finder = Finder()
    print(my_finder.search_rotated())