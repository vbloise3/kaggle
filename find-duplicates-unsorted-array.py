# How do you find duplicates from an unsorted array
import sys
class Dupremover(object):
    def __init__(self, theList = [1,3,2,5,4,6,1,7,2,3]):
        self.theList = theList
    def remove_dups(self):
        duplicated = set()
        not_duplicated = set()
        for item in self.theList:
            if item not in not_duplicated:
                not_duplicated.add(item)
            else:
                duplicated.add(item)
        return list(duplicated)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = list(map(int, arguments.split(",")))
        my_dup_remover = Dupremover(input)
    else:
        my_dup_remover = Dupremover()
    print(my_dup_remover.remove_dups())