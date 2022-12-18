from torch.utils.data.dataset import Dataset

label_dic = {  "cat" : 0, "dog" : 1 }

class MyCustomDataset(Dataset):
    def __init__(self, path):   # path 의미 없음 오류 방지용
        # data path
        self.all_data_path = "./image/*.jpg"
        pass
    def __getitem(self, index):
        image_path = self.all_data_path[index]
        # 결과 "image01.jpg, image02.jpg, image03.jpg,.... ""

        label_temp = image_path.split("\\")
        # 결과 [. , image , cat.jpg]
        label_temp = label_temp[2]
        label_temp = label_temp.replace(".jpg", "")
        label = label_dic[label_temp]
        # 결과 cat -> 0

        # image read
        # image = cv2.imread(image_path)
        image = 0        
        return (image, label)
    def __len__(self):
        return len(self.all_data_path)