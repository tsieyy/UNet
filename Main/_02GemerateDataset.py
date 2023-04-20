
# 生成数据集
import glob
import os, sys


LabelPaths = glob.glob(sys.path[0][:sys.path[0].rindex('\\')] + '\\OriLabelDataset/*.json')
# print(type(LabelPaths))
for LabelPath in LabelPaths:
	print(LabelPath)
	Name = 'data\\' + os.path.basename(LabelPath).split('.')[0]
	cmd = 'labelme_json_to_dataset {0} -o {1}'.format(LabelPath, Name)
	os.system(cmd) 
