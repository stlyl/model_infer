#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

class ort_infer
{
public:
	ort_infer(string model_path);
	cv::Mat detect(Mat srcimg);
private:
	std::vector<float> input_image_;
	int inpWidth;
	int inpHeight;
	int outWidth;
	int outHeight;
	int area;
	int seg_num_class;

	const bool keep_ratio = true;
	const int Cityscapes_COLORMAP[4][3] = { {0, 0, 0}, {0, 255, 0}, {255, 0, 0}, {0, 0, 255}};

	Ort::Env env = Env(ORT_LOGGING_LEVEL_ERROR, "DIS");
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

	std::vector<AllocatedStringPtr>In_AllocatedStringPtr;
	std::vector<AllocatedStringPtr>Out_AllocatedStringPtr;

	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	void normalize_(Mat img);
};