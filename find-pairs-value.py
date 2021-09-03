import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9], theValue = 8):
        self.theList = theList
        self.theValue = theValue
    def find_pairs(self):
        found_items = set()
        results = []
        for item in self.theList:
            if self.theValue - item in found_items:
                results.append([item, self.theValue - item])
            else:
                found_items.add(item)
        return results

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input = list(map(int, sys.argv[1].split(",")))
        value = int(sys.argv[2])
        my_finder = Finder(input, value)
    else:
        my_finder = Finder()
    print(my_finder.find_pairs())