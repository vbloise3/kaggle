import sys
class Ways(object):
    def __init__(self, theList = [5]):
        self.theList = theList
    def num_ways(self, num):
        n = num
        while n >= 0:
            if n == 0 or n == 1:
                return 1
            else:
                return self.num_ways(n - 1) + self.num_ways(n - 2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = int(sys.argv[1])
        my_ways = Ways(input)
    else:
        my_ways = Ways()
    print(my_ways.num_ways(input))