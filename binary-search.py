# How do you perform a binary search in a given array
import sys
class Sorter(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,78], value = 4):
        self.theList = theList
        self.value = value
    def search(self):
        low = 0
        high = len(self.theList) - 1
        return self.binary_search(self.theList, low, high, self.value)
    def binary_search(self, _list, low, high, x):
        if high >= low:
            mid = (high + low) // 2
            if _list[mid] == x:
                return mid
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif _list[mid] > x:
                return self.binary_search(_list, low, mid - 1, x)
            # Else the element can only be present in right subarray
            else:
                return self.binary_search(_list, mid + 1, high, x)
        else:
            return -1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        value = int(sys.argv[2])
        sorted_input = sorted(input)
        my_sorter = Sorter(sorted_input, value)
        position = my_sorter.search()
        print(input.index(sorted_input[position]))
    else:
        my_sorter = Sorter()
        print(my_sorter.search())