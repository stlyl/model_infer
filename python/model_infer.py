import os
import numpy as np
from PIL import Image

def out_to_rgb_np(out):
    CLASSES=('ignore','crack', 'spall', 'rebar')
    PALETTE=[[0, 0, 0],[0, 0, 255], [255, 0, 0], [0, 255, 0]]#RGB
    palette = np.array(PALETTE)
    assert palette.shape[0] == len(CLASSES)
    assert palette.shape[1] == 3
    assert len(palette.shape) == 2
    color_seg = np.zeros((out.shape[0], out.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[out == label, :] = color
    return color_seg

# 使用pytorch-gpu进行推理
def torch_infer():
    import torch
    import time
    from model import unet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet(4).to(device)
    model.load_state_dict(torch.load(r"weight/best_epoch_weights.pth"))
    model.eval()
    # 预处理
    image = Image.open(r"img/crack000680.jpg")
    image_data = np.array(image, dtype=np.float32)/255.0
    image_data  = np.expand_dims(np.transpose((image_data), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.to(device)
        pr = model(images)[0]
        pr = pr.argmax(axis=0)
    image_data =  np.uint8(pr.cpu())
    image_data = out_to_rgb_np(image_data)
    image = Image.fromarray(image_data)  # 将图像数据转换为 PIL 图像对象
    image.save("results/torch_infer_image.jpg") 
   

# 使用onnxruntime-gpu进行推理
def onnx_infer():
    import onnxruntime
    onnx_model_path = "weight/unet.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # 创建示例输入数据
    image = Image.open(r"img/crack000680.jpg")
    input_data = np.array(image, dtype=np.float32)/255.0
    input_data  = np.expand_dims(np.transpose((input_data), (2, 0, 1)), 0)
    # 运行推理
    output = ort_session.run(None, {"input": input_data})
    output = output[0].reshape((4,256, 256))
    output = np.argmax(output, axis=0)
    output = out_to_rgb_np(output)
    output = Image.fromarray(output)  # 将图像数据转换为 PIL 图像对象
    output.save("results/onnx_infer_image.jpg") 
    
def trt_infer():
    import os
    import cv2
    import time
    import tensorrt as trt
    import pycuda.driver as cuda  #GPU CPU之间的数据传输
    import pycuda.autoinit
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open("weight/sample.engine", "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    # 分配CPU锁页内存和GPU显存
    with engine.create_execution_context() as context:
        h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
    # 创建cuda流
    stream = cuda.Stream()
    image = Image.open(r"img/crack000680.jpg")
    image_data = np.array(image, dtype=np.float32)/255.0
    image_data  = np.expand_dims(np.transpose((image_data), (2, 0, 1)), 0)
    np.copyto(h_input, image_data.ravel())
    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # Run inference.
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output. 该数据等同于原始模型的输出数据
        h_output = h_output.reshape((4,256, 256))
        h_output = np.argmax(h_output, axis=0)
        h_output = h_output.astype(np.uint8)  # 将数据类型转换为 uint8
        h_output = np.squeeze(h_output) 
        h_output = out_to_rgb_np(h_output)
        image = Image.fromarray(h_output)  # 将图像数据转换为 PIL 图像对象
        image.save("results/trt_infer_image.jpg") 