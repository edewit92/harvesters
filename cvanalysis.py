import cv2
import matplotlib.pyplot as plt
import numpy as np

img_path = '/home/homer-dev/Documents/ebus_images/'
filename = '00000000_000001925313890F.png'

image = cv2.imread(img_path+filename)

plt.plot(np.mean(image[0,:,:], axis=1))

# # Display the image
# cv2.imshow("Image", image)

# # Wait for the user to press a key
# cv2.waitKey(0)

# # Close all windows
# cv2.destroyAllWindows()