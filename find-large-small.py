import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,4,5,6,7,8,9,0,-3,25]):
        self.theList = theList
    def get_largest_smallest(self):
        largest2 = None
        smallest2 = None
        largest = max(self.theList)
        smallest = min(self.theList)
        for item in self.theList:
            if (largest2 == None or largest2 < item) and item != largest: 
                largest2 = item 
            if (smallest2 == None or smallest2 > item) and item != smallest: 
                smallest2 = item
        return "largest: " + str(largest), "smallest: " +  str(smallest), "second smallest: " + str(smallest2), "second largest: " + str(largest2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        #input = [-3456,-21,5,89,103,-230]
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.get_largest_smallest())