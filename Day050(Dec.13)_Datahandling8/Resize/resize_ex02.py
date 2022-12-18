import os
from PIL import Image
from resize_ex01 import expand2square

def image_file(image_folder_path):
    all_root = []
    # print(",,", xml_folder_path)
    for (path, dir, files) in os.walk(image_folder_path):
        # print("...", path, dir, files)
        for filename in files:
            # image.png -> .png
            ext = os.path.splitext(filename)[-1]
            # print(ext)
            # ext_list = [".jpg",".png",".jpeg"]
            if ext ==".jpg" or ext ==".png":
                root = os.path.join(path, filename)
                # ./cvat_annotations/annotations.xml 로 저장 될거임
                all_root.append(root)
            else:
                # print("no image file...")
                continue
    return all_root

img_path_list = image_file("./image01/images/")
# print(img_path_list)

for img_path in img_path_list:
    # print(img_path)
    # image_name_temp = img_path.split("/")
    image_name_temp = os.path.basename(img_path)
    # # 경로 가져오는 다른방법
    # image_name_temp2 = os.path.abspath(img_path)   
    # print(image_name_temp2)
    image_name = image_name_temp.replace(".jpg", "")
    print(image_name)
    # kiwi_1
    
    img = Image.open("./images.jpeg")
    img_new = expand2square(img, (0,0,0)).resize((224,224))
    os.makedirs("./resize", exist_ok=True)
    img_new.save(f"./resize/{image_name}.png", quality=100)