test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")
print(utf8_encoded)
print(utf16_encoded)
print(utf32_encoded)
print(type(utf8_encoded))
# Get the byte values for the encoded string (integers from 0 to 255).
list(utf8_encoded)
# One byte does not necessarily correspond to one Unicode character!
print(len(test_string))

print(len(utf8_encoded))
