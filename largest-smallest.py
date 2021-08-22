# find the largest and smallest number in an unsorted integer array
import sys

def largest_smallest(theList):
    largest = theList[0]
    smallest = theList[0]
    secondLargest = None
    secondSmallest = None
    for item in theList:
        if item > largest:
            secondLargest = largest
            largest = item
        elif secondLargest == None or secondLargest < item:
            secondLargest = item
        if item < smallest:
            secondSmallest = smallest
            smallest = item
        elif secondSmallest == None or secondSmallest > item:
            secondSmallest = item
    print("largest: ", largest)
    print("second largest: ", secondLargest)
    print("smallest: ", smallest)
    print("second smallest ", secondSmallest)

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    largest_smallest(input)