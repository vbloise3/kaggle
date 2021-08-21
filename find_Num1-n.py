import sys
from collections import defaultdict

def find_missing_number(theList):
    my_dict = defaultdict(int)
    for item in theList:
        my_dict[item] += 1
    j = 1
    while j < len(theList):
        if j not in my_dict:
            return j
        j += 1

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int,arguments.split(",")))
    print(find_missing_number(input))