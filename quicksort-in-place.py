# How is an integer array sorted in place using the quicksort algorithm
import sys
class Sorter(object):
    def __init__(self, theList = [1,5,8,7,3,5,9,0,11,10]):
        self.theList = theList
    def quicksort_in_place(self):
        if len(self.theList) == 0:
            return
        self.quicksort(0, len(self.theList) - 1)
        return self.theList
    def quicksort(self, low, high):
        i = low
        j = high
        # Choose an element, called pivot, from the list or array. 
        # Generally pivot is the middle element of array
        pivot = self.theList[low + (high - low) // 2]
        # Reorder the list so that all elements with values less than the pivot 
        # come before the pivot, and all elements with values greater than the 
        # pivot come after it (equal values can go either way)
        while (i < j):
            while self.theList[i] < pivot:
                i += 1
            while self.theList[j] > pivot:
                j -= 1
            if i <= j:
                self.swap(i, j)
                i += 1
                j -= 1
            # Recursively apply the above steps to the sub-list of elements 
            # with smaller values and separately the sub-list of elements 
            # with greater values. If the array contains only one element 
            # or zero elements then the array is sorted
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