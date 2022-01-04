import math
def helper(input, kernel_size, stride, padding):
    return math.floor((input - kernel_size + 2 * padding)/stride + 1)

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h, kernel_size=1, stride=1, pad=0, dilation=1):
    h = math.floor( ((h + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)
    return h

print(conv_output_shape(31, 4, 1, 1))