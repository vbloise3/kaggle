import sys
class Calculator(object):
    def __init__(self, theList = [2,3,4,5,6,7,8,9], theValue = 9):
        self.theList = theList
        self.theValue = theValue
    def get_pairs(self):
        results = []
        items = set()
        for item in self.theList:
            if self.theValue - item in items:
                results.append([item, self.theValue - item])
            items.add(item)
        return results
if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_list = list(map(int, sys.argv[1].split(",")))
        input_value = int(sys.argv[2])
        my_calculator = Calculator(input_list, input_value)
    else:
        my_calculator = Calculator()
    print(my_calculator.get_pairs())