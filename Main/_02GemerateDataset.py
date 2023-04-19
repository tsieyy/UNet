
# 生成数据集
import glob
import os, sys

glob.glob(sys.path[0][:sys.path[0].rindex('\\')] + "\\dataset_raw\\OriImage\\*")
LabelPaths = glob.glob(sys.path[0][:sys.path[0].rindex('\\')] + '\\OriLabelDataset/*.json')

for LabelPath in LabelPaths:
	print(LabelPath)
	Name = os.path.basename(LabelPath).split('.')[0]
	cmd = 'labelme_json_to_dataset {0} -o {1}'.format(LabelPath, Name)
	os.system(cmd) 