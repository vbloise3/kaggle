import sys
# Find the missing number in the array
from collections import defaultdict
def find_sum_of_two(input, val):
 #TODO: Write - Your - Code
  found_values = set()
  for a in input:
    if val - a in found_values:
      return True

    found_values.add(a)
    
  return False

if __name__ == '__main__':
    # map(str, range(number + 1))
    # command line input: "1,2,3,4,6,7,8,9" 12
    instring = sys.argv[1]
    value = sys.argv[2]
    inputnumbers = map(int, instring.split(','))
    # print(inputnumbers)
    print(find_sum_of_two(inputnumbers, int(value)))