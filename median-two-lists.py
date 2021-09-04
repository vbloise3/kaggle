import sys
class Merger(object):
    def __init__(self, theFirstInput = [1,2,3,4,5], theSecondInput = [6,7,8,9,10]):
        self.theFirstInput = theFirstInput
        self.theSecondInput = theSecondInput
    def merge(self):
        merged_list = self.theFirstInput + self.theSecondInput
        merged_list.sort()
        n = len(merged_list)
        m = n - 1
        return(merged_list[n // 2] + merged_list[m // 2]) / 2

if __name__ == "__main__":
    if len(sys.argv) > 2:
        input1 = list(map(int, sys.argv[1].split(",")))
        input2 = list(map(int, sys.argv[2].split(",")))
        my_merger = Merger(input1, input2)
    else:
        my_merger = Merger()
    print(my_merger.merge())