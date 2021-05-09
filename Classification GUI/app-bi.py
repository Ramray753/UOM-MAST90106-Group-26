import os
import shutil
import sys

import numpy as np
from dearpygui import simple, core
from PIL import Image


def read_image(path, filename):
    with Image.open(os.path.join(path, filename)) as im:
        img = np.array(im).astype(int)
        ideal_shape = (1230, 410, 3)
        crop_bound = {
            "xmin": int((img.shape[1] - ideal_shape[1]) / 2),
            "xmax": int((img.shape[1] + ideal_shape[1]) / 2 - 1),
            "ymin": int((img.shape[0] - ideal_shape[0]) / 2),
            "ymax": int((img.shape[0] + ideal_shape[0]) / 2 - 1)
        }
        img_cropped = img[crop_bound["ymin"]:crop_bound["ymax"] + 1, crop_bound["xmin"]:crop_bound["xmax"] + 1]
        y_space = np.linspace(0, ideal_shape[0], 4).astype(int)
        y_space[-1] += 1
        return [img_cropped[y_space[i]:y_space[i + 1], ] for i in range(len(y_space) - 1)]

def clear_temp_folder():
    if os.path.isdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.mkdir(TEMP_DIR)

def clear_save_folder():
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.isfile(SAVE_FILE):
        with open(SAVE_FILE, "w") as file:
            file.write("image_name,crack_1,crack_2,crack_3")

def save_choice(sender, data):
    # Check availability
    if CROPPED[0] is None:
        core.set_main_window_title("Click 'LOAD NEXT' to load a new image!")
        return
    if not SHOW_NEXT[0]:
        core.set_main_window_title("Click 'SHOW NEXT' to show a new split image!")
        return

    # Write buffer
    LINE[0][-1] = data

    # Update window title
    core.set_main_window_title(f"Input Label: {data}")

def load_image():
    # Check availability
    if len(LINE[0]) < 4 or LINE[0][-1] is None:
        core.set_main_window_title("You haven't labeled all split images!")
        return
    if len(FILENAMES) == 0:
        core.set_main_window_title("★★★★★★★ ALL IMAGES ARE LABELED ★★★★★★★")
        FILE[0].write("\n" + ",".join(LINE[0]))
        FILE[0].close()
        core.delete_item("Manual Image Classifier")
        core.add_window("Manual Image Classifier", width=450, height=800, x_pos=0, y_pos=0)
        core.add_text("All images are labeled for current schedule!")
        core.add_text("Please close this window!")
        return

    # Load next image
    clear_temp_folder()
    CUR_FILENAME[0] = FILENAMES.pop(0)
    CROPPED[0] = read_image(PATH, CUR_FILENAME[0])

    # Update environment variable
    if FILE[0] is not None:
        FILE[0].write("\n" + ",".join(LINE[0]))
        FILE[0].close()
    FILE[0] = open(SAVE_FILE, "a")
    LINE[0].clear()
    LINE[0].append(CUR_FILENAME[0])
    SHOW_NEXT[0] = False

    # Update window title
    core.set_main_window_title(f"{TOTAL - len(FILENAMES)}/{TOTAL} Image loaded: {CUR_FILENAME[0]}")

def show_image():
    # Check availability
    if CROPPED[0] is None:
        core.set_main_window_title("No image is loaded! Please 'LOAD NEXT'!")
        return
    if LINE[0][-1] is None:
        core.set_main_window_title("Please label this image!")
        return
    if len(CROPPED[0]) == 0:
        core.set_main_window_title("This is the last split image! Please 'LOAD NEXT'!")
        return

    # Save last saved crack type to LINE buffer
    LINE[0].append(None)

    # Save split image in temporary folder
    im_array = CROPPED[0].pop(0)
    image = Image.fromarray(np.uint8(im_array))
    image_name = f"{CUR_FILENAME[0][:-4]}-Split-{3 - len(CROPPED[0])}.JPG"
    image_save_path = os.path.join(TEMP_DIR, image_name)
    image.save(image_save_path)

    # Update window title and image
    try:
        core.delete_item("temp")
    except:
        pass
    core.add_image("temp", image_save_path, parent="Manual Image Classifier", width=410, height=410)
    core.set_main_window_title(f"Now labeling: {image_name}")

    # Update environment variable
    SHOW_NEXT[0] = True


if __name__ == "__main__":
    # Path constants
    PATH = sys.argv[1]
    SAVE_DIR = os.path.join(PATH, "save")
    SAVE_FILE = os.path.join(SAVE_DIR, "labels.csv")
    TEMP_DIR = os.path.join(PATH, "temp")

    # Number of images to label
    label_number = int(sys.argv[2])
    last_index = -1

    # Set environment variables
    FILENAMES = sorted([filename for filename in os.listdir(PATH) if filename.endswith("JPG")])
    TOTAL = None
    CROPPED = [None]             # Container for each split image
    SHOW_NEXT = [False]          # Whether the first split image is shown for current full image
    CUR_FILENAME = [None]        # Current full image file name
    LINE = [[0, 0, 0, 0]]  # Buffer for saved labeling choices
    FILE = [None]                # Buffer for opened csv file

    # Make saving folder
    clear_temp_folder()
    clear_save_folder()
                  
    # Set window config
    core.set_main_window_size(450, 800)
    core.set_global_font_scale(1.25)
    core.set_style_window_padding(20, 20)

    # Add core buttons and text
    with simple.window("Manual Image Classifier", width=450, height=800, x_pos=0, y_pos=0):
        core.add_text("Click 'LOAD NEXT' to load next image.")
        core.add_text("Click 'SHOW NEXT' to show next split image.")
        core.add_text("Click the type of crack for that split image.")
        core.add_spacing(count=4)
        core.add_separator()
        core.add_spacing(count=4)
        core.add_text("Crack Type: ")
        core.add_same_line()
        core.add_button("NO", callback=save_choice, callback_data="none")
        core.add_same_line()
        core.add_button("YES", callback=save_choice, callback_data="cracked")
        core.add_spacing(count=4)
        core.add_separator(name="sep1")
        core.add_spacing(count=4)
        core.add_text("Operations: ")
        core.add_same_line()
        core.add_button("LOAD NEXT", callback=load_image)
        core.add_same_line()
        core.add_button("SHOW NEXT", callback=show_image)
        core.add_spacing(count=4)
        core.add_separator(name="sep2")
        core.add_spacing(count=4)
        with open(SAVE_FILE, "r") as file:
            record = file.readlines()
            if len(record) == 1:
                core.add_text("No previous images are labeled! ")
            else:
                last_record = record[-1].split(",")[0]
                last_index = FILENAMES.index(last_record)
                core.add_text(f"Last image labeled: {last_record}")
        if last_index < len(FILENAMES) - 1:
            FILENAMES = FILENAMES[(last_index + 1): min(last_index + 1 + label_number, len(FILENAMES))]
            TOTAL = len(FILENAMES)
            core.add_text(f"{len(FILENAMES)} Images to be labeled: {FILENAMES[0]} - {FILENAMES[-1]}")
        else:
            FILENAMES.clear()
            core.add_text("All images are labeled!")
        core.add_spacing(count=4)
        core.add_separator()
        core.add_spacing(count=4)

    # Check if all images are labeled
    if len(FILENAMES) == 0:
        core.delete_item("Manual Image Classifier")
        core.add_window("Manual Image Classifier", width=450, height=800, x_pos=0, y_pos=0)
        core.add_text("All images are labeled in this folder!")
        core.add_text("Please close this window!")

    core.start_dearpygui()
