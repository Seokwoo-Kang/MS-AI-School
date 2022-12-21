import cv2
import random
import albumentations as A

# Define a function to visualize an image
def visualize(image):
    cv2.imshow("visualization", image)
    cv2.waitKey(0)

# Load the image from the disk
image = cv2.imread("./Agumentation/weather.jpg")

# # visualize the original image
# visualize(image)
# # print("image szie", image.shape)
# # # image szie (337, 600, 3)

# RandomRain
transform = A.Compose([
    # A.RandomRain(brightness_coefficient=0.2, drop_width=4, blur_value=2, p=1)
    # A.RandomSnow(brightness_coeff=2, snow_point_lower=0.9, snow_point_upper=0.9, p=1)
    # A.RandomSunFlare(flare_roi=(0,0,1,0.5), angle_lower=0.3, p=1)
    # A.RandomShadow(shadow_roi=(0,0.5,1,1),num_shadows_lower=3,num_shadows_upper=3,shadow_dimension=8,p=1)
    A.RandomFog(fog_coef_lower= 0.3, fog_coef_upper= 1, alpha_coef=0.5, p=1)
])
transformed = transform(image=image)
visualize(transformed["image"])