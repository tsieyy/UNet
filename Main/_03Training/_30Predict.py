
import torch, os
from Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _21CalEvaluationIndicator import *
from PIL import Image
Device = torch.device('cpu')

def read_images(path): # 读取图像
	files = os.listdir(path)
	img_files =[]
	for file in files:
		index = file.find('.')
		prefix = file[index+1:]
		if prefix in ['jpg', 'png']:
			img_files.append(file)
	return img_files

# %% 载入数据、模型
# FolderPath = '/home/cxq/workspace2/2019.10.23PipeEdgeDetecion/2019.10.23LossFunctionTest/Test/Dataset'
FolderPath = '../Dataset'
UnconvertedPath = './Output/Unconverted'
ConvertedPath = './Output/Converted'
MaskPath = './Output/Mask'

TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, 1)

predict_dataset = PredictDataset(os.path.join(FolderPath, 'Predict'), PredictImgTransform)
PredictDatasetLoader = DataLoader(dataset=predict_dataset, batch_size=1, drop_last=False, num_workers=0, pin_memory=True)
# Unet_BCELoss_Adam
SaveFolder = 'Output'
Unet = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(Device)
Unet.load_state_dict(torch.load(os.path.join(SaveFolder, '0300.pt'), map_location=Device))

# %% 测试
Unet.eval()  # 评估模式
torch.set_grad_enabled(False)
OutputS = []        # 存储检测数据，用于指标计算
LabelS = []
################################################################
# print(PredictDatasetLoader)
# for img, sample_name in PredictDatasetLoader:
# 	print(sample_name)
# 	InputImg = img.float().to(Device)
# 	OutputImg = Unet(InputImg)
# 	Output = OutputImg.cpu().numpy()[0]
# 	# 生成效果图
# 	OutputImg = OutputImg.cpu().numpy()[0, 0]
# 	OutputImg = (OutputImg*255).astype(np.uint8)
# 	Input = img.numpy()[0][0]
# 	Input = (Normalization(Input) * 255).astype(np.uint8)
# 	ResultImg = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
# 	ResultImg[...,2] = OutputImg
# 	plt.show()
# 	cv2.imwrite(os.path.join(ConvertedPath, sample_name[0][:4] + '.png'), ResultImg)
################################################################
# print(ValDataLoader)
# for Iter, (Input, Label, SampleName) in enumerate(ValDataLoader):
# 	end = timer(8)
# 	print(SampleName)
# 	InputImg = Input.float().to(Device)
# 	OutputImg = Unet(InputImg)
# 	Output = OutputImg.cpu().numpy()[0]
# 	Label = Label.detach().cpu().numpy()[0]
# 	OutputS.append(Output)
# 	LabelS.append(Label)
# 	end('5555')

# 	# 生成效果图
# 	OutputImg = OutputImg.cpu().numpy()[0, 0]
# 	OutputImg = (OutputImg*255).astype(np.uint8)
# 	Input = Input.numpy()[0][0]
# 	Input = (Normalization(Input) * 255).astype(np.uint8)
# 	ResultImg = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
# 	ResultImg[...,2] = OutputImg
# 	plt.show()
# 	cv2.imwrite(os.path.join(UnconvertedPath, SampleName[0] + '.png'), ResultImg)




# Input, Label, SampleName = next(iter(ValDataLoader))
# print(SampleName)
Input, SampleName = next(iter(PredictDatasetLoader))

InputImg = Input.float().to(Device)
outputImg = Unet(InputImg)
output = outputImg.cpu().numpy()[0, 0]
OutputImg = (output * 255).astype(np.uint8)

Input = Input.numpy()[0][0]
Input = (Normalization(Input) * 255).astype(np.uint8)
print(Input.shape)
plt.imshow(Input)
plt.show()
ResultImg = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
plt.imshow(ResultImg)
plt.show()
print(ResultImg.shape)
ResultImg[...,0] = OutputImg
plt.imshow(ResultImg)
plt.show()
print(ResultImg.shape)


# print(outputImg)
# print(output)
