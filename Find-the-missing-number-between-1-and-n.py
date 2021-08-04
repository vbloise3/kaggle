import sys
# Find the missing number in the array of numbers up to n 
from collections import defaultdict
def find_missing(limit):
 #TODO: Write - Your - Code
  print("input: ", input)
  theotherlist = defaultdict(int)
  for i in input:
      theotherlist[i] += 1
      print (theotherlist)
  j = 1
  while j < 10:
      if j not in theotherlist:
          return j
      j+=1
  return -1

if __name__ == '__main__':
    # map(str, range(number + 1))
    # command line input: 1000
    instring = sys.argv[1]
    inputnumbers = map(int, instring.split(','))
    # print(inputnumbers)
    print(find_missing(inputnumbers))