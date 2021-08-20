import sys

def getDups(inList):
    _size = len(inList)
    repeated = []

    for i in range(_size):
        k = i +1
        for j in range(k, _size):
            if inList[i] == inList[j] and inList[i] not in repeated:
                repeated.append(inList[i])
    return(repeated)

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(getDups(input))