import os
import glob

image_folder_path ='./iamge'
image_path = glob.glob(os.path.join(image_folder_path, ".png"))

# for path in image_path:
#     file_name = os.path.basename(path)
#     if "dark" in file_name:
