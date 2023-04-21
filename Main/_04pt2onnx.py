# .pt 文件转换成 .onnx文件进行tensorrt推理
import os.path

import numpy as np
import torch.onnx
from _03Training._03Unet import UNet
import onnx
import onnxruntime as ort


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, 3, 128, 128, requires_grad=True)

    # Export the model   
    torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "unet.onnx",       # where to save the model
        export_params=True,  # store the trained parameter weights inside the model ile 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names=['modelInput'],   # the model's input names
        output_names=['modelOutput'],  # the model's output names
        # dynamic_axes={'modelInput': {0: 'batch_size'},    # variable length axes
        #                 'modelOutput': {0: 'batch_size'}}
                      )
    print(" ") 
    print('Model has been converted to ONNX')


if __name__ == "__main__": 


    model = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(device)

    path = r"/home/bad/0500.pt"

    model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    # Conversion to ONNX
    if not os.path.exists('unet.onnx'):
        Convert_ONNX()

    print('start check onnx')
    net = onnx.load("unet.onnx")
    test_arr = torch.randn(1, 3, 128, 128).numpy().astype(np.float32)
    ort_session = ort.InferenceSession('unet.onnx')
    outputs = ort_session.run(None, {'input': test_arr})

    print('onnx result:', outputs[0])
