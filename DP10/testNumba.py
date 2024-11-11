import numpy as np
import time

# # 普通的 Python 函数
# def compute_square_sum(arr):
#     result = 0.0
#     for x in arr:
#         result += x ** 2
#     return result

# # 创建一个大数组
# arr = np.random.rand(1000000)

# # 测量运行时间
# start_time = time.time()
# print("Sum of squares:", compute_square_sum(arr))
# print("Execution time without Numba:", time.time() - start_time, "seconds")


from numba import jit


# 使用 Numba 的 JIT 编译器加速函数
@jit(nopython=True)
def compute_square_sum(arr):
    result = 0.0
    for x in arr:
        result += x ** 2
    return result

# 创建一个大数组
arr = np.random.rand(1000000)

# 测量运行时间
start_time = time.time()
print("Sum of squares with Numba:", compute_square_sum(arr))
print("Execution time with Numba:", time.time() - start_time, "seconds")

# 第二次运行
arr = np.random.rand(1000000)
start_time = time.time()
print("Sum of squares with Numba:", compute_square_sum(arr))
print("Execution time with Numba:", time.time() - start_time, "seconds")
