from PIL import Image
# pip install Pillow 파이썬에서 사용하는 이미지 라이브러리 cv2와 같음

def expand2square(pil_img, background_color) :
    width, height = pil_img.size
    # print(width, height)
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # image add (추가할 이미지, 붙일 위치 (가로, 세로))
        result.paste(pil_img, (0,(width-height)//2))
        return result
    else :
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height-width)//2, 0))
        return result

# cv2.imread()
img = Image.open("./images.jpeg")

# expand2square(img, background_color=None)
# print(img)
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=174x290 at 0x1DCB8F4A1D0>

img_new = expand2square(img, (0,0,0)).resize((224,224))
img_new.save("./test.png", quality=100)