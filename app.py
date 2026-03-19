import numpy as np

# Specify the path to your .npy file
file_path = 'D:\\MLOPS\\artifact\\18_03_2026_14_04_17\\data_transformation\\transformed\\test.npy' 

# Load the data
data_array = np.load(file_path)

# You can now use the loaded data like any other NumPy array
print(data_array)
print(f"Shape of the array: {data_array.shape}")
print(f"Data type of the array: {data_array.dtype}")
