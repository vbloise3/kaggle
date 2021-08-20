import sys

def get_largest_smallest(theList):
    largest = theList[0]
    smallest = theList[0]
    largest2 = None
    smallest2 = None
    for item in theList[1:]:
        if item > largest:
            largest2 = largest
            largest = item
        elif largest2 == None or largest2 < item:
            largest2 = item
        if item < smallest:
            smallest2 = smallest
            smallest = item
        elif smallest2 == None or smallest2 > item:
            smallest2 = item
    print("largest: ", largest)
    print("second largest: ", largest2)
    print("smallest: ", smallest)
    print("second smallest: ", smallest2)

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    get_largest_smallest(input)