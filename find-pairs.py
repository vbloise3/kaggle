# find all pairs of an integer array whose sum is equal to a given number
import sys

def find_pairs(theList, value):
    pairs = []
    results = set()
    for item in theList:
        if value - item in results:
            pairs.append([value-item, item])
        results.add(item)
    return(pairs)

if __name__ == "__main__":
    argument_list = sys.argv[1]
    argument_value = sys.argv[2]
    input = list(map(int, argument_list.split(",")))
    print(find_pairs(input, int(argument_value)))