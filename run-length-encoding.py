# aaabbccc -> 3a2b3c
import sys
from collections import defaultdict
class Encoder(object):
    def __init__(self, theString = 'aaabbccc'):
        self.theString = theString
    def encode(self):
        my_dict = defaultdict(int)
        result = ''
        for element in self.theString:
            my_dict[element] += 1
        for key in my_dict:
            result += key + str(my_dict[key])
        return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = sys.argv[1]
        my_encoder = Encoder(input)
    else:
        my_encoder = Encoder()
    print(my_encoder.encode())