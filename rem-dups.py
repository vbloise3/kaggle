import sys
class Remover(object):
    def __init__(self, theList = [1,1,2,3,4,3,5,6,7,12,8,5,5,7,9,4,8,1,10,10,10,10,10,10,11]):
        self.theList = theList
    def remove_dups(self):
        _size = len(self.theList)
        duplicates = set()
        non_duplicates = set()
        for i in range(_size):
            k = i + 1
            for j in range(k, _size):
                if self.theList[i] == self.theList[j]:
                    duplicates.add(self.theList[j])
                else:
                    non_duplicates.add(self.theList[j])
        return sorted(non_duplicates) #sorted(non_duplicates.union(duplicates))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = list(map(int, sys.argv[1].split(",")))
        my_remover = Remover(input)
    else:
        my_remover = Remover()
    print(my_remover.remove_dups())