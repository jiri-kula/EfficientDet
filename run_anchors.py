# %%
from model.anchors import Anchors
import numpy as np

# %%
an = Anchors()
tensor = an.get_anchors(320, 320)

# Convert the tensor to a NumPy array
tensor_values = tensor.numpy()

# Open a text file in write mode
with open("anchors.txt", "w") as file:
    # Write the values to the text file
    np.savetxt(file, tensor_values, fmt="%s")

print("Tensor values have been written to tensor_values.txt")
# %%
