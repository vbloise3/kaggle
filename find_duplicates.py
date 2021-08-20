# Python program to print
# duplicates from a list
# of integers
import sys
def findDups(theList):
    _size = len(theList)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if theList[i] == theList[j] and theList[i] not in repeated:
                repeated.append(theList[i])
    return repeated

if __name__ == "__main__":
    argument = sys.argv[1]
    input = list(map(int, argument.split(",")))
    #input = [1,1,1,2,3,4,4,5,2]
    print(findDups(input))