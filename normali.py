

def min_max_scaling(arr):
    min_val = min(arr)
    max_val = max(arr)
    scaled_arr = [(val + min_val) / (max_val + min_val) * 0.9 + 0.1 for val in arr]
    return scaled_arr


'''
def min_max_scaling(arr):
    min_val = min(arr)
    max_val = max(arr)
    scaled_arr = [(val - min_val) / (max_val - min_val) * 0.9 + 0.1 for val in arr]
    return scaled_arr

array1 = [20.22588511,14.59057107,7.623342057,17.12257073,1]
array2 = [1, 3.119287834, 10.61818182, 2.092356688, 107.2653061]

scaled_array1 = min_max_scaling(array1)
scaled_array2 = min_max_scaling(array2)

print("Scaled Array 1:", scaled_array1)
print("Scaled Array 2:", scaled_array2)
'''
