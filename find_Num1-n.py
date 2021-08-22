# find missing number in an array of ints from 1-n
import sys

def find_missing(theList):
    found = set()
    _size = len(theList)
    for item in theList:
        found.add(item)
    j = 1
    while j < _size:
        if j not in found:
            return j
        j += 1
    return -1

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(find_missing(input))