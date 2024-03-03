# 0. python_api

- [x] 使用pytorch推理
- [x] 使用onnxruntime推理
- [x] 使用tensorrt推理

------

# 1. cpp_api

- [x] cv::dnn
- [x] onnxruntime
- [ ] tensorrt

## 1.1 Quick start

```shell
git clone https://github.com/stlyl/model_infer.git
cd model_infer
# 修改CMakeLists.txt--17,改为你的ort路径
set(ONNXRUNTIME_ROOT_PATH /mnt/d/linux_project/your_onnxruntime_path/)
mkdir build
cd build
cmake ..
make 
cd ..
./bin/image_onnx 
```