import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,8,9,10,11,12,6,7,-1,-2,-3,-4,-5]):
        self.theList = theList

    def get_longest_sequence(self):
        longest_sequence = 0
        for item in self.theList:
            current_sequence = 1
            current_num = item
            while current_num + 1 in self.theList:
                current_sequence += 1
                current_num += 1
            longest_sequence = max(current_sequence, longest_sequence)
        return longest_sequence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.get_longest_sequence())