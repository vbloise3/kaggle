import sys

def find_dups(theList):
    _size = len(theList)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if theList[i] == theList[j]:
                repeated.append(theList[i])
    return repeated

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(find_dups(input))