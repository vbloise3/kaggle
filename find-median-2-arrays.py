import sys
class Finder(object):
    def __init__(self, firstList = [1,2,3,4,5,6], secondList = [7,8,9,10,11,12]):
        self.firstList = firstList
        self.secondList = secondList
    def find_median(self):
        combined = self.firstList + self.secondList
        _size = len(combined)
        n = _size // 2
        m = n - 1
        return (combined[n] + combined[m]) / 2

if __name__ == "__main__":
    if len(sys.argv) > 2:
        list1 = list(map(int, sys.argv[1].split(",")))
        list2 = list(map(int, sys.argv[2].split(",")))
        my_finder = Finder(list1, list2)
    else:
        my_finder = Finder()
    print(my_finder.find_median())