# Arrays store elements in contiguous memory locations, 
# resulting in easily calculable addresses for the elements stored 
# and this allows a faster access to an element at a specific index. 
# Linked lists are less rigid in their storage structure and elements 
# are usually not stored in contiguous locations, hence they need to 
# be stored with additional tags giving a reference to the next element. 
# This difference in the data storage scheme decides which data structure 
# would be more suitable for a given situation. 

# Major differences are listed below: 

#   Size:  Since data can only be stored in contiguous blocks of memory 
#      in an array, its size cannot be altered at runtime due to risk of 
#      overwriting over other data. However in a linked list, each node 
#      points to the next one such that data can exist at scattered 
#      (non-contiguous) addresses; this allows for a dynamic size which 
#      can change at runtime.
#   Memory allocation: For arrays at compile time and at runtime for 
#      linked lists. but, dynamically allocated array also allocates memory 
#      at runtime.
#   Memory efficiency: For the same number of elements, linked lists use 
#      more memory as a reference to the next node is also stored along 
#      with the data. However, size flexibility in linked lists may make 
#      them use less memory overall; this is useful when there is 
#      uncertainty about size or there are large variations in the size of 
#      data elements; memory equivalent to the upper limit on the size 
#      has to be allocated (even if not all of it is being used) while 
#      using arrays, whereas linked lists can increase their sizes 
#      step-by-step proportionately to the amount of data.
#   Execution time: Any element in an array can be directly accessed with 
#      its index; however in case of a linked list, all the previous 
#      elements must be traversed to reach any element. Also, better cache 
#      locality in arrays (due to contiguous memory allocation) can 
#      significantly improve performance. As a result, some operations 
#      (such as modifying a certain element) are faster in arrays, 
#      while some other (such as inserting/deleting an element in the data) 
#      are faster in linked lists.