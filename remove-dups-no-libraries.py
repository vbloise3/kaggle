# Remove duplicates from an array without using any collections library
import sys
class Remover(object):
    def __init__(self, theList = [1,2,3,2,5,4,5,6,7,8,6]):
        self.theList = theList
    def remove_dups(self):
        self.theList.sort()
        result = []
        previous = self.theList[0]
        result.append(previous)
        for i in range(len(self.theList)):
            item = self.theList[i]
            if previous != item:
                result.append(item)
            previous = item
        return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        my_remover = Remover(input)
    else:
        my_remover = Remover()
    print(my_remover.remove_dups())