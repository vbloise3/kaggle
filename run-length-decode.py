# 2a3b1c -> aabbbc
import sys
class Decoder(object):
    def __init__(self, theString = '2a3b1c'):
        self.theString = theString
    def decode(self):
        string_list = []
        interim_result = ''
        result = ''
        multiplier = ''
        character = ''
        reverse = []
        numbers = ['0','1','2','3','4','5','6','7','8','9']
        for element in self.theString:
            string_list.append(element)
        for j in range(len(string_list) - 1, -1, -1):
            if string_list[j] not in numbers:
                character = string_list[j]
            else:
                multiplier += string_list[j]
                if string_list[j - 1] in numbers:
                    multiplier += string_list[j]
                else:
                    interim_result = multiplier + character
                    multiplier = ''
            #result += interim_result
            if interim_result != '':
                reverse.append(interim_result)
            interim_result = ''
        for element in reversed(reverse):
            i = 0
            while i < int(element[:-1]):
                result += element[-1]
                i += 1
        return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input = str(sys.argv[1])
        #input = '2a3b1c'
        my_decoder = Decoder(input)
    else:
        my_decoder = Decoder()
    print(my_decoder.decode())