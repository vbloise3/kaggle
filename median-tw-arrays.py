# How to find a median of two sorts arrays
import sys
class Sorter(object):
    def __init__(self, firstList = [1,2,4,56,7,3], secondList = [4,5,6,7,84,6]):
        self.firstList = firstList
        self.secondList = secondList
    def find_median(self):
        merged_list = self.firstList + self.secondList
        merged_list.sort()
        n = len(merged_list)
        m = n - 1
        return (merged_list[n // 2] + merged_list[m // 2]) / 2.0
if __name__ == "__main__":
    if len(sys.argv) > 2:
        arguments = sys.argv
        input1 = list(map(int, arguments[1].split(",")))
        input2 = list(map(int, arguments[2].split(",")))
        my_sorter = Sorter(input1, input2)
    else:
        my_sorter = Sorter()
    print(my_sorter.find_median())