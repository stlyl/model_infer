#include "ort_infer.h"


ort_infer::ort_infer(string model_path)
{
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		AllocatorWithDefaultOptions allocator;
		In_AllocatedStringPtr.push_back(ort_session->GetInputNameAllocated(i, allocator));
		input_names.push_back(In_AllocatedStringPtr.at(i).get());
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		AllocatorWithDefaultOptions allocator;
		Out_AllocatedStringPtr.push_back(ort_session->GetOutputNameAllocated(i, allocator));
		output_names.push_back(Out_AllocatedStringPtr.at(i).get());
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->outHeight = output_node_dims[0][2];
	this->outWidth = output_node_dims[0][3];
	this->area = this->inpHeight*this->inpWidth;
	this->seg_num_class = output_node_dims[0][1];
}


Mat ort_infer::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void ort_infer::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}


Mat ort_infer::detect(Mat srcimg)
{
	Mat seg_img = srcimg.clone();
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(),
		input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	const float* pseg = ort_outputs[0].GetTensorMutableData<float>();
	float ratioh = (float)newh / srcimg.rows;
	float ratiow = (float)neww / srcimg.cols;
	int n = 0,  i = 0, j = 0;
	for (i = 0; i < seg_img.rows; i++)//这是外层循环，遍历输出图像 seg_img 的行。
	{
		for (j = 0; j < seg_img.cols; j++)//这是内层循环，遍历输出图像 seg_img 的列。
		{
			const int x = int(j*ratiow) + padw;//根据当前列索引 j、水平方向的缩放比例 ratiow 和填充值 padw 计算出原始图像中的对应列坐标 x。
			const int y = int(i*ratioh) + padh;
			float max_prob = -1;//初始化最大概率值为 -1，用于保存当前像素点所属类别的最大概率值。
			int max_ind = 0;//初始化最大概率对应的类别索引为 0，用于保存当前像素点所属的类别。
			for (n = 0; n < seg_num_class; n++)//这是一个循环，遍历所有可能的类别。
			{
				float pix_data = pseg[n * area + y * this->inpWidth + x];//从分割结果数据中获取特定位置 (x, y) 处属于第 n 类的像素值。
				if (pix_data > max_prob)
				{
					max_prob = pix_data;
					max_ind = n;
				}
			}
			seg_img.at<Vec3b>(i, j)[0] = this->Cityscapes_COLORMAP[max_ind][0];
			seg_img.at<Vec3b>(i, j)[1] = this->Cityscapes_COLORMAP[max_ind][1];
			seg_img.at<Vec3b>(i, j)[2] = this->Cityscapes_COLORMAP[max_ind][2];
		}
	}
	Mat combine;
	if (srcimg.rows < srcimg.cols)
	{
		vconcat(srcimg, seg_img, combine);
	}
	else
	{
		hconcat(srcimg, seg_img, combine);
	}
	return combine;
}

