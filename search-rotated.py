import sys

def find_target(theList, target):
    low = 0
    high = len(theList)
    while low < high:
        mid = low + (high - low) // 2
        if target == theList[mid]:
            return mid
        if theList[low] < theList[mid]:
            if target >= theList[low] and target < theList[mid]:
                high = mid
            else:
                low = mid + 1
        else:
            if target <= theList[high - 1] and target > theList[mid]:
                low = mid + 1
            else:
                high = mid
    return -1

if __name__ == "__main__":
    arguments_list = sys.argv[1]
    arguments_target = sys.argv[2]
    input = list(map(int, arguments_list.split(",")))
    target = int(arguments_target)
    print(find_target(input, target))