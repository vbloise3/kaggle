import sys
class Sorter(object):
    def __init__(self, theList = [4,3,6,2,7,5,9,8,2]):
        self.theList = theList
    def quicksort_in_place(self):
        self.quicksort(0, len(self.theList) - 1)
        return self.theList
    def quicksort(self, low, high):
        i = low
        j = high
        pivot = self.theList[low + (high - low) // 2]
        # reorder around the pivot
        while i < j:
            while self.theList[i] < pivot:
                i += 1
            while self.theList[j] > pivot:
                j -= 1
            if i <= j:
                self.swap(i,j)
                i += 1
                j -= 1
        # recursively apply reorder
            if low < j:
                self.quicksort(low, j)
            if i < high:
                self.quicksort(i, high)
    def swap(self, i, j):
        temp = self.theList[i]
        self.theList[i] = self.theList[j]
        self.theList[j] = temp

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_sorter = Sorter(input)
    else:
        my_sorter = Sorter()
    print(my_sorter.quicksort_in_place())