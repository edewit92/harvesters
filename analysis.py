import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
data = np.load('images/image_data_20241003_161240.npy')
image = data[50]

# Extract the R, G, B channels
R = image[:, :, 0]  # Red channel
G = image[:, :, 1]  # Green channel
B = image[:, :, 2]  # Blue channel

# Calculate the overall intensity (mean across R, G, B)
intensity_mean = np.mean(image[0,:,:], axis=1)

# Optionally, you could calculate the overall intensity using the Euclidean norm
# intensity_norm = np.sqrt(R**2 + G**2 + B**2)

# Call the function to plot the intensity lineout
plt.plot(R[0], color='red')
plt.plot(G[0], color='green')
plt.plot(B[0], color='blue')

# Show all plots
plt.tight_layout()
plt.show()
