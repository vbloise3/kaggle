# How do you search a target value in a rotated array
import sys

def find_target_location(theList, target):
    low = 0
    high = len(theList)
    while low < high:
        mid = low + (high - low) // 2
        if theList[mid] == target:
            return mid
        if theList[low] <= theList[mid]:
            if target >= theList[low] and target < theList[mid]:
                high = mid
            else:
                low = mid + 1
        else:
            if target <= theList[high-1] and target > theList[mid]:
                low = mid + 1
            else:
                high = mid
    return -1

if __name__ == "__main__":
    argument_list = sys.argv[1]
    argument_target = sys.argv[2]
    input = list(map(int, argument_list.split(",")))
    target = int(argument_target)
    print(find_target_location(input, target))