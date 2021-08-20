import sys
from collections import defaultdict

def findNum(inList):
    my_dict = defaultdict(int)
    for i in inList:
        my_dict[i] +=1
    j = 1
    while j < len(inList):
        if j not in my_dict:
            return j
        j +=1
    return -1

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    print(findNum(input))