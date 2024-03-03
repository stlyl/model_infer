#include "ort_infer.h"

int main()
{
	string model_path = "rcfile/unet.onnx";
	ort_infer my_model(model_path);
	string imgpath = "rcfile/spall000648.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = my_model.detect(srcimg);
	imwrite("rcfile/ort_mask.jpg",outimg);
}