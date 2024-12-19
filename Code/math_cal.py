import math
#这个文件包含了一些数学运算，就两个，频率到音分转换，音分到频率转换
def calculate_c(f, f_ref=10):
    C = 1200 * math.log2(f / f_ref)
    return C
def calculate_f(C):
    f = 2**(C/1200)*10
    return f

print(calculate_c(215)/20)
print(calculate_c(218)/20)

