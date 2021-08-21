import sys
# Find all pairs of ints that add up to given value
def find_all_sum_of_two(input, val):
  theResult = []
  found_values = set()
  for a in input:
    if val - a in found_values:
      #return True
      theResult.append([val-a,a])
    found_values.add(a)
  #return False
  return theResult

if __name__ == '__main__':
    # command line input: "1,2,3,4,6,7,8,9" 12
    instring = sys.argv[1]
    value = sys.argv[2]
    inputnumbers = list(map(int, instring.split(',')))
    print(find_all_sum_of_two(inputnumbers, int(value)))