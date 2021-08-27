# How to convert a byte array to String
import sys
class Converter(object):
    def __init__(self, theList = bytearray("test", encoding="utf-8")):
        self.theList = theList
    def convert_byte_array_to_string(self):
        return bytes(self.theList)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arguments = sys.argv[1]
        input = arguments
        bytes_input = bytearray(input, encoding="utf-8")
        my_converter = Converter(bytes_input)
    else:
        my_converter = Converter()
    print(my_converter.convert_byte_array_to_string())