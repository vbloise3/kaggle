import sys
class Fibonacci(object):
    def __init__(self, theValue = 6):
        self.theValue = theValue
    def get_sequence(self, theValue):
        if theValue == 0:
            return 0
        elif theValue == 1:
            return 1
        else:
            return self.get_sequence(theValue - 1) + self.get_sequence(theValue - 2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = int(sys.argv[1])
        my_fib = Fibonacci(input)
    else:
        my_fib = Fibonacci()
    for i in range(input):
        print(my_fib.get_sequence(i))