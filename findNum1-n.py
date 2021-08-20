import sys
from collections import defaultdict

def findMissingInt(theList):
    s = defaultdict(int)
    for i in theList:
        s[i] += 1
    j = 1
    while j < len(s):
        if j not in s:
            return j
        j+=1
    return -1


if __name__ == "__main__":
    argument = sys.argv[1]
    input = list(map(int, argument.split(",")))
    print(findMissingInt(input))