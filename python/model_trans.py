import torch
from model import unet

def torch2onnx():
    model = unet(n_classes = 4).cuda()
    model.load_state_dict(torch.load(r"weight/best_epoch_weights.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.randn(1, 3, 256, 256,device=device)
    torch.onnx.export(model, dummy_input, "weight/unet.onnx", opset_version=11, verbose=False,
        input_names = ["input"], output_names=["output"])

def onnxtotrt():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(r"weight/unet.onnx")
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass # Error handling code here
    
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    serialized_engine = builder.build_serialized_network(network, config)
    with open("weight/sample.engine", "wb") as f:
        f.write(serialized_engine)