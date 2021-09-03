# aaabbccc -> 3a2b3c
import sys
from collections import defaultdict
class Encoder(object):
    def __init__(self, theString = 'aaabbccca'):
        self.theString = theString
    def encode(self):
        result = []
        result_string = ''
        string_list = []
        character = ''
        multiplier = 0
        for element in self.theString:
            string_list.append(element)
        i = len(string_list) - 1
        while i > 0:
            character = string_list[i]
            multiplier = 1
            while string_list[i - 1] == character:
                multiplier += 1
                i -= 1
            interim_result = str(multiplier) + character
            result.append(interim_result)
            multiplier = 0
            i -= 1
        for element in reversed(result):
            result_string += element 
        return result_string

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = sys.argv[1]
        my_encoder = Encoder(input)
    else:
        my_encoder = Encoder()
    print(my_encoder.encode())