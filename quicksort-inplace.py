import sys
class Sorter(object):
    def __init__(self, theList = [1,2,3,5,6,7,8,3,5]):
        self.theList = theList
    def quicksort_in_place(self):
        if len(self.theList) == 0:
            return
        self.quicksort(0, len(self.theList) - 1)
        return self.theList
    def quicksort(self, low, high):
        i = low
        j = high
        pivot = self.theList[low + (high - low) // 2]
        while i < j:
            while self.theList[i] < pivot:
                i += 1
            while self.theList[j] > pivot:
                j -= 1
            if i <= j:
                self.swap(i, j)
                i += 1
                j -= 1
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
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        my_sorter = Sorter(input)
    else:
        my_sorter = Sorter()
    print(my_sorter.quicksort_in_place())