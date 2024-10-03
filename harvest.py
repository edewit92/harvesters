"""
TODO
- Display stream diagnostics
- Debug performance issues 10GigE vs 1GigE
- Get real-time trigger
- Timestamps
"""

from harvesters.core import Harvester
from genicam.gentl import TimeoutException, GenericException
import cv2
import numpy as np
import time
from pynput import keyboard
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

WIDTH = 4096  # Image buffer width
HEIGHT = 1  # Image buffer height
PIXEL_FORMAT = "RGB8"  # Camera pixel format

SDK_CTI_PATH = "/opt//mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti"  # Input camera SDK .cti file
CAMERA_MODEL = "SW-4000TL-10GE" # Camera product model

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

    ia.num_buffers = 100
    apply_settings_to_camera(ia, settings)

    return h, ia

def shutdown_camera(image_acquirer, harvester):
    """Shutdown the camera and free up resources"""
    image_acquirer.stop()
    image_acquirer.destroy()
    harvester.reset()

def save_image(image, file_name, frame_idx):
    """Save a single image to a file"""
    image_path = f"{file_name}_{frame_idx}.png"
    cv2.imwrite(image_path, image)

def acquire_frames(ia, settings):
    """Acquire a number of frames based on settings"""
    print(f"\n[INFO] Acquiring {settings.n_frames} frames.\n")

    # Initialize matplotlib figure and axis for displaying the image
    if settings.show_video:
        # Initialize timestamp to control the display frame rate at 30 FPS
        last_frame_time = time.time()

        fig, ax = plt.subplots()
        line, = ax.plot(np.zeros(WIDTH))  # Initialize the line plot with zeros
        ax.set_xlim(0, WIDTH)  # Set x-axis range to width of the image
        ax.set_ylim(0, 255)  # Set y-axis range to 0-255 (pixel intensity range)
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('Mean Intensity')
        plt.ion()  # Enable interactive mode
        plt.show()  # Ensure the plot is rendered first

    frames = np.zeros([settings.n_frames, HEIGHT, WIDTH, 3], dtype=np.uint8)
    number_dropped = 0
    payload = frames.size

    ia.start()
    actime1 = time.time()

    for i in range(settings.n_frames):
        print(f'\rAcquiring frame {i+1}')

        try:
            buffer = ia.try_fetch(timeout=1/settings.fps)
            if buffer is None:
                print(f"Buffer fetch timeout at frame {i+1}")
                number_dropped = number_dropped + 1
                continue
            component = buffer.payload.components[0]
            image = component.data.reshape(HEIGHT, WIDTH, 3)
            np.copyto(frames[i], image)

            if settings.show_video: # Update the plot if video display is enabled
                current_time = time.time()
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame >= 1/30: # Time-based frame rate control (30 FPS)
                    line.set_ydata(np.mean(image[0, :, :], axis=1))
                    plt.draw()  # Redraw the updated plot
                    last_frame_time = current_time
                    plt.pause(0.001)

        except ValueError as e:
            print(f"[ERROR] Reshaping failed on frame {i+1}: {e}")
            number_dropped = number_dropped + 1
        except TimeoutException:
            print(f"[WARNING] Frame {i+1} was dropped or timed out.")
            number_dropped = number_dropped + 1
        except Exception as e:
            print(f"[ERROR] Unexpected error during acquisition: {e}")
        finally:
            if buffer is not None: buffer.queue()
    
    ia.stop()
    actime2 = time.time()

    theoretical_bandwidth = payload/(settings.n_frames - number_dropped) * settings.fps
    utilised_bandwidth = payload/(actime2-actime1)
    print("\nAcquisition statistics:")
    print(f"Frames dropped: {number_dropped/settings.n_frames*100} %")
    print(f"Theoretical bandwidth: {theoretical_bandwidth*8/1e6} Mbps")
    print(f"Utilised bandwidth: {utilised_bandwidth*8/1e6} Mbps")

    # Close the plot window after the acquisition ends
    if settings.show_video:
        plt.ioff()  # Disable interactive mode
        plt.show()

    return frames

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

        print(f"[INFO] Camera exposure time set to {settings.exposure_time} µs")
        print(f"[INFO] Camera FPS set to {settings.fps}")
        # The number of frames is handled by the application, not typically by the camera
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

if __name__ == "__main__":

    file_name = 'images/video'
    
    # Initialize the settings object
    settings = AcquisitionSettings()

    h, ia = None, None  # Initialize h and ia to None before the try block
    try:
        
        h, ia = init_camera(settings)
        #list_available_nodes(ia.data_streams[0].node_map)

        while True:
            
            print("\nOptions:")
            print("1. Acquire images")
            print("2. Change settings")
            print("3. Quit")
            user_input = input("Choose an option (1/2/3): ")

            if user_input == '1':
                frames = acquire_frames(ia, settings)
                np.save('image_data.npy', frames)
            elif user_input == '2':
                settings_menu(ia, settings)
            else:
                print("\n[INFO] Shutting down.")
                break

    except Exception as e:
        print(f"[CRITICAL ERROR] Something went wrong: {e}")
    finally:
        # Ensure camera shutdown
        if ia is not None and h is not None:
            shutdown_camera(ia, h)
            print("Camera safely shut down.")
