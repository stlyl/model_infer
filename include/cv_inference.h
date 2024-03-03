#ifndef _CV_INFERENCE_H_
#define _CV_INFERENCE_H_
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

using namespace cv;
using namespace std;

class cv_infer
{
public:
	cv_infer(string model_path);
	cv::Mat detect(Mat srcimg);
private:
	dnn::Net net;
    const int inpWidth = 256;
    const int inpHeight = 256;
};
#endif