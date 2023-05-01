
import torch, os
from _05Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _21CalEvaluationIndicator import *
from PIL import Image
Device = torch.device("cuda:0")

def read_images(path): # 读取图像
    files = os.listdir(path)
    img_files =[]
    for file in files:
        index = file.find('.')
        prefix = file[index+1:]
        if prefix in ['jpg', 'png']:
            img_files.append(file)
    return img_files

# 载入数据、模型
# FolderPath = '/home/cxq/workspace2/2019.10.23PipeEdgeDetecion/2019.10.23LossFunctionTest/Test/Dataset'
FolderPath = '../../Dataset'
UnconvertedPath = './Output/Unconverted'
ConvertedPath = './Output/Converted'
MaskPath = './Output/Mask'
TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PipeDatasetLoader(FolderPath, 1)
# Unet_BCELoss_Adam
SaveFolder = 'Output'
Unet = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast = True, ActivateFunLast = torch.sigmoid).to(Device)
Unet.load_state_dict(torch.load(os.path.join(SaveFolder, '0300.pt'), map_location = Device))

# 测试
Unet.eval()  # 评估模式
torch.set_grad_enabled(False)
OutputS = []        # 存储检测数据，用于指标计算
LabelS = []
for Iter, (Input, Label, SampleName) in enumerate(ValDataLoader):
	end = timer(8)
	print(SampleName)
	InputImg = Input.float().to(Device)
	OutputImg = Unet(InputImg)
	Output = OutputImg.cpu().numpy()[0]
	Label = Label.detach().cpu().numpy()[0]
	OutputS.append(Output)
	LabelS.append(Label)
	end('5555')
	# 生成效果图
	OutputImg = OutputImg.cpu().numpy()[0, 0]
	OutputImg = (OutputImg*255).astype(np.uint8)
	Input = Input.numpy()[0][0]
	Input = (Normalization(Input) * 255).astype(np.uint8)
	ResultImg = cv2.cvtColor(Input, cv2.COLOR_GRAY2RGB)
	ResultImg[...,2] = OutputImg
	plt.show()
	Mask = Label[0]

	cv2.imwrite(os.path.join(UnconvertedPath, SampleName[0] + '.png'), ResultImg)
	#cv2.imwrite(os.path.join(MaskPath, SampleName[0] + '.png'), MaskImg)
	# image_array是归一化的二维浮点数矩阵
	# Mask *= 255  # 变换为0-255的灰度值
	# MaskImg = Image.fromarray(Mask)
	# Mask = Label[0]
	# MaskImg = MaskImg.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
	# MaskImg.save(os.path.join(MaskPath, SampleName[0] + '.png'))
	# 上面这个方法可以保存图像,但会导致指标计算错误

OutputFlatten = np.vstack(OutputS).ravel()
LabelFlatten = np.vstack(LabelS).ravel()
# ROC, AUC
fpr, tpr, AUC = ROC_AUC(LabelFlatten, OutputFlatten, ShowROC = True)
print('AUC:', AUC)
# POC, AP
recall, precision, MF, AP = PRC_AP_MF(LabelFlatten, OutputFlatten, ShowPRC = True)
# mIOU = iou_mean(LabelFlatten, OutputFlatten, n_classes=1)
# print(mIOU)
print('MF:', MF)
print('AP:', AP)
plt.show()