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
      return self.data

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

# merge sorted linked lists 
def merge_sorted(head1, head2):
  # if both lists are empty then merged list is also empty
  # if one of the lists is empty then other is the merged list
  if head1 == None:
    return head2
  elif head2 == None:
    return head1

  mergedHead = None
  if head1.data <= head2.data:
    mergedHead = head1
    head1 = head1.next
  else:
    mergedHead = head2
    head2 = head2.next

  mergedTail = mergedHead
  
  while head1 != None and head2 != None:
    temp = None
    if head1.data <= head2.data:
      temp = head1
      head1 = head1.next
    else:
      temp = head2
      head2 = head2.next

    mergedTail.next = temp
    mergedTail = temp

  if head1 != None:
    mergedTail.next = head1
  elif head2 != None:
    mergedTail.next = head2

  return mergedHead
    
if __name__ == '__main__':
    # map(str, range(number + 1))
    # command line input: "10,11,12,13,14,15,18,23" "2,3,4,5,6,7,19,20"
    firstlist = sys.argv[1]
    secondlist = sys.argv[2]
    l_1 = list(map(int, firstlist.split(',')))
    l_2 = list(map(int, secondlist.split(',')))
    # print(l_1)
    print(merge_sorted([4, 8, 15, 19, 22],[7, 9, 10, 16]))