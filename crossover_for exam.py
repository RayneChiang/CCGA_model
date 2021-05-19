import numpy as np
import math
import random


String_A = '11111110000111111'
String_B = '00011111110001111'
String_A = '001110'
String_B = '001001'
two_list = []
for i in range(len(String_A)):
    for j in range(i+1, len(String_A)):
        child_A = String_A[0:i] + String_B[i:j] + String_A[j:]
        child_B = String_B[0:i] + String_A[i:j] + String_B[j:]
        two_list.append(child_A)
        two_list.append(child_B)

print(len(np.unique(two_list)))

one_list = []
for i in range(len(String_A)):
    child_A = String_A[0:i] + String_B[i:]
    child_B = String_B[0:i] + String_A[i:]
    one_list.append(child_A)
    one_list.append(child_B)

print(len(np.unique(one_list)))
