import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the scrambled matrix
scrambled = np.load("scrambled.npy")   # make sure scrambled.npy is in the same folder

print("Original shape:", scrambled.shape)
print("Total elements:", scrambled.size)

# Step 2: Reshape into square
n = int(np.sqrt(scrambled.size))
decoded = scrambled.reshape(n, n)

# Step 3: Fix orientation
# Try these ONE BY ONE until the image looks correct:
# decoded = np.flipud(decoded)        # flip up-down
# decoded = np.fliplr(decoded)        # flip left-right
# decoded = decoded.T                 # transpose
# decoded = np.rot90(decoded, k=1)    # rotate 90°

# Step 4: Show the final image
plt.imshow(decoded, cmap="gray")
plt.axis("off")
plt.show()
