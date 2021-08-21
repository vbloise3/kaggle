import sys

def get_pairs(theList, val):
    pairs = []
    found = set()
    for item in theList:
        if val - item in found:
            pairs.append([val-item, item])
        found.add(item)
    return pairs

if __name__ == "__main__":
    _list = sys.argv[1]
    value = sys.argv[2]
    list_of_ints = list(map(int, _list.split(",")))
    print(get_pairs(list_of_ints, int(value)))