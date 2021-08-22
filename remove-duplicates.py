# How to remove duplicates from a given array
import sys

def remove_duplicates(theList):
    non_dups = set()
    duplicates = set()
    _size = len(theList)
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if theList[i] == theList[j]:
                duplicates.add(theList[j])
            else:
                non_dups.add(theList[j])
    return(sorted(non_dups.union(duplicates)))

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(remove_duplicates(input))