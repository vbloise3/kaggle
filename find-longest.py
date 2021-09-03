import sys
class Finder(object):
    def __init__(self, theList = [1,2,3,5,6,7,8,-1,-2,-3,0]):
        self.theList = theList
    def longest_streak(self):
        longest = 0
        for num in self.theList:
            current_streak = 1
            current_num = num
            while current_num + 1 in self.theList:
                current_num += 1
                current_streak += 1
            longest = max(longest, current_streak)
        return longest

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_finder = Finder(input)
    else:
        my_finder = Finder()
    print(my_finder.longest_streak())