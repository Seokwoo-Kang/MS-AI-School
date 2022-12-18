from torch.utils.data.dataset   import Dataset
import os, glob
import pandas as pd

class MyCustomDataSet(Dataset):
    def __init__(self, path):
        # Data path <- 'C:\\MS AI School\\Venv\\Pytorch'
        self.data_path = glob.glob(os.path.join(path, '*.csv'))
        csv_path = self.data_path[0]
        df = pd.read_csv(csv_path)
        print(df)
        self.filename = df.iloc[:,1].values
        self.x1 = df.iloc[:,2].values
        self.y1 = df.iloc[:,3].values
        self.w = df.iloc[:,4].values
        self.h = df.iloc[:,5].values

    def __getitem__(self, index):
        file_name = self.filename[index]
        x1 = self.x1[index]
        y1 = self.x1[index]
        w = self.x1[index]
        h = self.x1[index]
        return file_name, x1, y1, w, h

    def __len__(self):
        return len(self.data_path)

data = MyCustomDataSet('C:\\MS AI School\\Venv\\Pytorch')

for item in data:
    print(item)
