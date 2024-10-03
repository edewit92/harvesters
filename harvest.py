"""
TODO
- Debug performance issues 10GigE vs 1GigE
- Get real-time trigger
"""

import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt
from harvesters.core import Harvester
from genicam.gentl import TimeoutException, GenericException
import cv2
import numpy as np
import time

import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

# -------------------------------------------------------------------------------
# CLASSES
# -------------------------------------------------------------------------------

class AcquisitionSettings:
    def __init__(self, n_frames=100, fps=100, exposure_time=10, show_video=False):
        self.n_frames = n_frames
        self.fps = fps
        self.exposure_time = exposure_time
        self.show_video = show_video

    def display(self):
        print(f"Frames: {self.n_frames}, FPS: {self.fps}, Exposure: {self.exposure_time} µs")

# -------------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------------

def init_camera(settings):
    """Initialize the camera and configure settings"""
    print("\n[INFO] Connecting to camera, please wait...\n")

    h = Harvester()
    h.add_file(SDK_CTI_PATH)
    h.update()

    print(h.device_info_list)

    try:
        ia = h.create({'model': 'SW-4000TL-10GE'})
    except Exception as e:
        print(f"[ERROR] Camera is busy or not connected: {str(e)}")
        exit(1)

    ia.num_buffers = 32
    apply_settings_to_camera(ia, settings)

    return h, ia

def shutdown_camera(image_acquirer, harvester):
    """Shutdown the camera and free up resources"""
    try:
        if image_acquirer is not None:
            image_acquirer.stop()  # Safely stop acquisition
            image_acquirer.destroy()  # Release resources
        if harvester is not None:
            harvester.reset()  # Reset the harvester
    except Exception as e:
        print(f"[ERROR] Shutdown failed: {e}")

def save_image(image, file_name, frame_idx):
    """Save a single image to a file"""
    image_path = f"{file_name}_{frame_idx}.png"
    cv2.imwrite(image_path, image)

def acquire_frames(ia, settings):
    """Acquire a number of frames based on settings"""
    print(f"\n[INFO] Acquiring {settings.n_frames} frames.\n")

    # For live plotting
    if settings.show_video:
        # Initialize timestamp to control the display frame rate at 30 FPS

        last_frame_time = time.perf_counter()  # Higher precision
        fig, ax = plt.subplots()
        line, = ax.plot(np.zeros(WIDTH))    # Initialize the line plot with zeros
        ax.set_xlim(0, WIDTH)               # Set x-axis range to width of the image
        ax.set_ylim(0, 255)                 # Set y-axis range to 0-255 (pixel intensity range)
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Mean Intensity')

        # Enable interactive mode
        plt.ion()
        plt.show()

    # For data storage
    n_frames = settings.n_frames
    frames = np.zeros([n_frames, HEIGHT, WIDTH, 3], dtype=np.uint8)
    timestamps = np.zeros(n_frames, dtype=np.uint64)  # Assuming timestamps are integers

    # For statistics
    number_dropped = 0
    payload = frames.size / n_frames * 8 # bits / frame
    actime1 = time.time()
    previous_timestamp = None  # To store the timestamp of the previous buffer

    ia.start()
    
    for i in range(settings.n_frames):

        try:

            buffer = ia.try_fetch(timeout=10/settings.fps)
            if buffer is None:
                print(f"Buffer fetch timeout at frame {i+1}")
                number_dropped += 1
                continue

            component = buffer.payload.components[0]
            try:
                image = component.data.reshape(HEIGHT, WIDTH, 3)
            except ValueError as ve:
                # Print a detailed error message including the frame number
                print(f"[ERROR] Value error during reshaping at frame {i+1}: {ve}")
                number_dropped += 1
                continue

            current_timestamp = buffer.timestamp_ns

            if previous_timestamp is not None and (i + 1) % 10 == 0:
                    time_diff = current_timestamp - previous_timestamp
                    bandwidth = (image.size*8/1e6)/(time_diff/1e9) #Mbps
                    print(f"\rAcquired frame {i+1}: Time diff: {time_diff/1e3:.2f} us, Bandwidth: {bandwidth:.2f} Mbps", end='', flush=True)

            previous_timestamp = current_timestamp # Update previous timestamp

            # Update the plot if video display is enabled at reduced FPS
            current_time = time.perf_counter()
            if settings.show_video and current_time - last_frame_time >= 1/30: 
                # Calculate mean intensity along one axis
                intensity_mean = np.mean(image[0, :, :], axis=1)
                
                # Update the line's data with the new intensity values
                line.set_ydata(intensity_mean)  # Update data

                # Redraw the plot
                plt.draw()
                plt.pause(0.01)  # A small pause to allow GUI updates
                
                last_frame_time = current_time

            # Copy image data and store timestamp
            np.copyto(frames[i], image)
            timestamps[i] = current_timestamp
                    
        except TimeoutException as te:
            print(f"[ERROR] Frame acquisition timed out: {te}")
            number_dropped += 1
        except GenericException as e:
            print(f"[ERROR] Unexpected error during acquisition: {e}")
        finally:
            if buffer is not None: buffer.queue()
    
    ia.stop()

    # Print final statistics
    actime2 = time.time()
    theoretical_bandwidth = payload * settings.fps / 1e6
    average_bandwidth = payload * (n_frames - number_dropped) / (actime2-actime1) / 1e6

    print("\n\nAcquisition statistics:")
    print(f"Frames dropped: {number_dropped/settings.n_frames*100:.2f} %")
    print(f"Theoretical bandwidth: {theoretical_bandwidth:.2f} Mbps")
    print(f"Utilised bandwidth: {average_bandwidth:.2f} Mbps")

    # Save data
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S") # Timestamp for file name
    image_filename = f"images/image_data_{timestamp_str}.npy"
    timestamp_filename = f"images/timestamps_{timestamp_str}.csv"
    np.save(image_filename, frames)
    np.savetxt(timestamp_filename, timestamps, delimiter=',', fmt='%d', header='Timestamps')

    # Close the plot window
    if settings.show_video:
        plt.ioff()  # Disable interactive mode
        plt.close(fig)
        
def settings_menu(ia, settings):
    """Modify acquisition settings"""
    while True:
        print("\nSettings Menu:")
        print(f"1. Change number of frames (current: {settings.n_frames})")
        print(f"2. Change FPS (current: {ia.remote_device.node_map.AcquisitionLineRate.value})")
        print(f"3. Change exposure time (current: {ia.remote_device.node_map.ExposureTime.value} µs)")
        print(f"4. Toggle video display (current: {settings.show_video})")
        print("5. Back to main menu")

        choice = input("Choose an option (1/2/3/4/5): ")

        if choice == '1':
            try:
                settings.n_frames = int(input("Enter the new number of frames: "))
            except ValueError:
                print("[ERROR] Invalid input. Please enter a valid number.")

        elif choice == '2':
            try:
                settings.fps = int(input("Enter the new FPS: "))
                apply_settings_to_camera(ia, settings)
            except ValueError:
                print("[ERROR] Invalid input. Please enter a valid number.")

        elif choice == '3':
            try:
                settings.exposure_time = int(input("Enter the new exposure time: "))
                apply_settings_to_camera(ia, settings)
            except ValueError:
                print("[ERROR] Invalid input. Please enter a valid number.")

        elif choice == '4':
            settings.show_video = not settings.show_video
            print(f"[INFO] Show video set to {settings.show_video}")

        elif choice == '5':
            break

        else:
            print("[ERROR] Invalid option. Please choose 1, 2, 3, or 4.")

def apply_settings_to_camera(ia, settings):
    """Apply the settings to the camera node map."""
    try:

        ia.remote_device.node_map.ExposureTime.value = settings.exposure_time
        ia.remote_device.node_map.AcquisitionLineRate.value = settings.fps

        print("\nDevice settings:")
        print(f"[INFO] Camera exposure time set to {settings.exposure_time} µs")
        print(f"[INFO] Camera FPS set to {settings.fps}")
        print(f"[INFO] Number of frames for acquisition set to {settings.n_frames}")     

    except Exception as e:
        print(f"\n[ERROR] Failed to apply settings to camera: {e}\n")

def list_available_nodes(node_map):

    print("\nAvailable Nodes in the Node Map:\n")
    try:
        # Use the dir() method to list all available nodes and properties
        available_nodes = dir(node_map)
        print(available_nodes)
                
    except Exception as e:
        print(f"[ERROR] Failed to list nodes: {e}")

# -------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------

WIDTH = 4096  # Image buffer width
HEIGHT = 1  # Image buffer height
PIXEL_FORMAT = "RGB8"  # Camera pixel format

SDK_CTI_PATH = "/opt//mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"  # Input camera SDK .cti file
CAMERA_MODEL = "SW-4000TL-10GE" # Camera model

if __name__ == "__main__":

    file_name = 'images/video'
    settings = AcquisitionSettings()
    h, ia = None, None

    try:
        
        h, ia = init_camera(settings)

        while True:
            
            print("\nOptions:")
            print("1. Acquire images")
            print("2. Change settings")
            print("3. Quit")
            user_input = input("Choose an option (1/2/3): ")

            if user_input == '1':
                acquire_frames(ia, settings)
            elif user_input == '2':
                settings_menu(ia, settings)
            elif user_input == '3':
                print("\n[INFO] Shutting down.")
                break
            else:
                print("\n[INFO] Invalid input.")
                continue

    except Exception as e:
        print(f"[CRITICAL ERROR] Something went wrong: {e}")
    finally: # Ensure camera shutdown
        shutdown_camera(ia, h)

