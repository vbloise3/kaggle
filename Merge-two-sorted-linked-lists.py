import sys
# A single node of a singly linked list
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

# merge sorted linked lists 
def mergelinkedlists(l_1, l_2):
    LL1 = LinkedList()
    LL2 = LinkedList()
    for i in l_1:
        LL1.insertAtEnd(i)
    for i in l_2:
        LL2.insertAtEnd(i)
    # if both lists are empty then merged list is also empty
    # if one of the lists is empty then other is the merged list
    if LL1.head == None:
        return LL2
    elif LL2.head == None:
        return LL1
    mergedTail = LinkedList()
    if LL1.head.data <= LL2.head.data:
        mergedTail.insertAtEnd(LL1.head.data)
        LL1.head = LL1.head.next
    else:
        mergedTail.insertAtEnd(LL2.head.data)
        LL2.head = LL2.head.next
    while LL1.head != None and LL2.head != None:
        temp = None
        if getData(LL1.head) <= getData(LL2.head):
            temp = LL1.head
            LL1.head = LL1.head.next
        else:
            temp = LL2.head
            LL2.head = LL2.head.next
        mergedTail.insertAtEnd(temp.data)
        #mergedTail = temp
    if LL1.head != None:
        mergedTail.insertAtEnd(LL1.head.data)
    elif LL2 != None:
        mergedTail.insertAtEnd(LL2.head.data)
    return mergedTail
    
if __name__ == '__main__':
    # map(str, range(number + 1))
    # command line input: "10,11,12,13,14,15,18,23" "2,3,4,5,6,7,19,20"
    firstlist = sys.argv[1]
    #firstlist = "10,11,12,13,14,15,18,23"
    secondlist = sys.argv[2]
    #secondlist = "2,3,4,5,6,7,19,20"
    l_1 = map(int, firstlist.split(','))
    l_2 = map(int, secondlist.split(','))
    # print(inputnumbers)
    theMergedList = mergelinkedlists(l_1, l_2)
    theMergedList.traverse_list()