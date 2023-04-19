
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

InputImgSize=(128,128)
# %% 训练过程图片的变换
TrainImgTransform = transforms.Compose([
	# transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.5, 2.), shear=10),
	# transforms.RandomHorizontalFlip(),
	# transforms.RandomVerticalFlip(),

	#随机裁剪到同一个大小、宽高比
	transforms.RandomResizedCrop(InputImgSize, scale=(1., 1.), interpolation=Image.BILINEAR),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
TrainLabelTransform = transforms.Compose([
	# transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.5, 2.), shear=10),
	# transforms.RandomHorizontalFlip(),
	# transforms.RandomVerticalFlip(),
	transforms.RandomResizedCrop(InputImgSize, scale=(1., 1.), interpolation=Image.NEAREST),
	# transforms.RandomResizedCrop(InputImgSize, scale=(1., 1.)),
	transforms.ToTensor(),
])

# %% 测试过程图片变换
ValImgTransform = transforms.Compose([
	transforms.Resize(InputImgSize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
ValLabelTransform = transforms.Compose([
	transforms.Resize(InputImgSize, interpolation=Image.NEAREST),
	transforms.ToTensor(),
])

# TODO:预测过程图片变换
# 预测过程图片变换
PredictImgTransform = transforms.Compose([
	transforms.Resize(InputImgSize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10])
])

class PipeDataset(Dataset):
	def __init__(self, DatasetFolderPath, ImgTransform, LabelTransform, ShowSample=False):
		self.DatasetFolderPath = DatasetFolderPath
		self.ImgTransform = ImgTransform
		self.LabelTransform = LabelTransform
		self.ShowSample = ShowSample
		self.SampleFolders = os.listdir(self.DatasetFolderPath)

	def __len__(self):
		return len(self.SampleFolders)

	def __getitem__(self, item):
		SampleFolderPath = os.path.join(self.DatasetFolderPath, self.SampleFolders[item])  # 样本文件夹路径
		FusionImgPath = os.path.join(SampleFolderPath, 'img.png')
		LabelImgPath = os.path.join(SampleFolderPath, 'label.png')
		FusionImg = Image.open(FusionImgPath)
		LabelImg = Image.open(LabelImgPath)
		LabelImg = np.array(LabelImg)*255
		LabelImg = Image.fromarray(LabelImg)

		# %% 保证样本和标签具有相同的变换
		seed = np.random.randint(2147483647)
		random.seed(seed)
		FusionImg = self.ImgTransform(FusionImg)
		random.seed(seed)
		LabelImg = self.LabelTransform(LabelImg)

		# %% 显示Sample
		if self.ShowSample:
			plt.figure(self.SampleFolders[item])
			Img = FusionImg.numpy()[0]
			Label = LabelImg.numpy()[0]
			Img = (Normalization(Img) * 255).astype(np.uint8)
			Label = (Normalization(Label) * 255).astype(np.uint8)
			Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
			Img[..., 2] = Label
			plt.imshow(Img)
			plt.show()
		return FusionImg, LabelImg, self.SampleFolders[item]



class PredictDataset(Dataset):
	def __init__(self, dataset_folder, img_transform):
		self.dataset_folder = dataset_folder
		self.img_transform = img_transform
		self.sample_folders = os.listdir(self.dataset_folder)

	def __len__(self):
		return len(self.sample_folders)

	def __getitem__(self, item):
		img_path = os.path.join(self.dataset_folder, self.sample_folders[item])  # 样本路径
		img = Image.open(img_path)
		seed = np.random.randint(2147483647)
		random.seed(seed)
		img = self.img_transform(img)
		return img, self.sample_folders[item]





def PipeDatasetLoader(FolderPath, BatchSize=1, ShowSample=False):
	TrainFolderPath = os.path.join(FolderPath, 'Train')
	TrainDataset = PipeDataset(TrainFolderPath, TrainImgTransform, TrainLabelTransform, ShowSample)
	TrainDataLoader = DataLoader(TrainDataset, batch_size=BatchSize, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
	ValFolderPath = os.path.join(FolderPath, 'Val')
	ValDataset = PipeDataset(ValFolderPath, ValImgTransform, ValLabelTransform, ShowSample)
	ValDataLoader = DataLoader(ValDataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
	# TODO:读取PredictDataset
	# PredictFolderPath = os.path.join(FolderPath, 'Predict')
	# PredictDataset = PipeDataset(PredictFolderPath, PredictImgTransform, )
	return TrainDataset, TrainDataLoader, ValDataset, ValDataLoader


def Normalization(Array):  # 数组归一化到0~1
	min = np.min(Array)
	max = np.max(Array)
	if max - min == 0:
		return Array
	else:
		return (Array - min) / (max - min)


if __name__ == '__main__':
	FolderPath = '../Dataset'
	# TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, BatchSize=1, ShowSample=True)
	# for epoch in range(1):
	# 	for i, (Img, Label, SampleName) in enumerate(TrainDataLoader):
	# 		print(SampleName)
	# 		print(Img.shape)
	# 		print(Label.max())
	predictDataset = PredictDataset(os.path.join(FolderPath, 'Predict'), PredictImgTransform)
	for i in range(1):
		img, sample = predictDataset[1]
		print(img.shape)
		print(sample)
