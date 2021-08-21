import sys
from collections import defaultdict
import random

testList = {0: "find missing number in an array of ints from 1-n",
1: "find duplicates in a list of Numbers",
2: "find the largest and smallest number in an unsorted integer array",
3: "find all pairs of an integer array whose sum is equal to a given number"}

tests = defaultdict(int)

class Node:
  # constructor
  def __init__(self, data = None, next=None): 
    self.data = data
    self.next = next
  def getNext(self):
    return self.next

# A Linked List class with a single head node
class LinkedList:
  def __init__(self):  
    self.head = None

  def getdata(self):
      return self.head.data

  def insertAtEnd(self,data):
        temp=Node(data)
        if self.head==None:
            self.head=temp
        else:
            curr=self.head
            while curr.next!=None:
                curr=curr.next
            curr.next=temp

  def __contains__(self,data):
    if self.head == None:
        return False
    else:
        p = self.head
        while p is not None:
            if p.data == data:
                return True
            p = p.next
        return False            

  def traverse_list(self):
      if self.head is None:
          print("List has no element")
          return
      else:
          n = self.head
          while n is not None:
              print(n.data , " ")
              n = n.next

# Helper function
def getData(node):
    return node.data  

def printLL(list):
  node = list.head
  while node:
    print(node.data, ' ')
    node = node.next

def getTest(input):
    random_Number = random.randint(0,input)
    #print("random number: ", random_Number)
    tests[random_Number] += 1
    if tests[random_Number] < 2:
        return(testList[random_Number])
    else:
        return getTest(input)


def get_random_test_list(theList):
    LL1 = LinkedList()
    for i in range(theList[0]):
        j = i + 1
        if testList[j] not in LL1:
            LL1.insertAtEnd(getTest(j))
    LL1.traverse_list()

if __name__ == "__main__":
    arguments = sys.argv[1]
    input = list(map(int, arguments.split(",")))
    #input = [2]
    print(get_random_test_list(input))