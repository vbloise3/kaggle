# find duplicates in a list of Numbers
import sys

def find_dups(theList):
    _size = len(theList)
    dups = set()
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if theList[i] == theList[j] and theList[j] not in dups:
                dups.add(theList[j])
    return(sorted(dups))

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(find_dups(input))