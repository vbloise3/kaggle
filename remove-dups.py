# remove dups from array of ints
import sys

def find_dups(theList):
    _size = len(theList)
    duplicates = set()
    non_duplicates = set()
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if theList[i] == theList[j]:
                duplicates.add(theList[j])
            else:
                non_duplicates.add(theList[j])
    return(sorted(duplicates.union(non_duplicates)))

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(find_dups(input))