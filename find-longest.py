import sys
class Sequencer:
    def __init__(self, theList = [1,2,3,4,5,7,8,9,45,6]):
        self.theList = theList
    def get_longest(self):
        longest_streak = 0
        for num in self.theList:
            current_num = num
            current_streak = 1
            while current_num + 1 in self.theList:
                current_num += 1
                current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        return longest_streak

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    my_sequencer = Sequencer(input)
    print(my_sequencer.get_longest())