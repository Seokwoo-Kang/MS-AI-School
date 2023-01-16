
batch_size=128
lr=0.001
criterion = LabelSmoothingCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=0.001) # 모델의 파라메터를 준다. 그리고 0.001이 기본이다.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2,threshold_mode='abs',min_lr=1e-9, verbose=True)
save_dir = "./models/"
num_epoch = 50



train_transform = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.Resize(height=224, width=224),
    A.ToGray(p=1),
    A.Rotate(60, p=1),
    A.Rotate(45, p=1),
    A.Rotate(30, p=1),
    A.Rotate(15, p=1),
    A.RandomShadow(p=0.4),
    A.RandomFog(p=0.4),
    A.RandomSnow(p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.ShiftScaleRotate(shift_limit=5, scale_limit=0.05,
                       rotate_limit=15, p=0.7),
    A.VerticalFlip(p=1),
    A.HorizontalFlip(p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])