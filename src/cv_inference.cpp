#include "cv_inference.h"


cv_infer::cv_infer(string model_path)
{
    this->net = dnn::readNetFromONNX(model_path);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

cv::Mat cv_infer::detect(Mat srcimg)
{
    Mat blob;
    dnn::blobFromImage(srcimg, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), 
        Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    Mat outs;
    this->net.forward(outs);
    std::cout << "Dimension information of outs:" << std::endl;
    cout << outs.size[0] << "x" << outs.size[1] << "x" << outs.size[2] << endl;
    std::vector<std::string> CLASSES = { "ignore", "crack", "spall", "rebar" };
    std::vector<cv::Vec3b> PALETTE = { cv::Vec3b(0, 0, 0), cv::Vec3b(255, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(0, 0, 255) }; // bgr
    cv::Mat color_seg(outs.rows, outs.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < outs.rows; row++)
    {
        for (int col = 0; col < outs.cols; col++)
        {
            float label = outs.at<float>(row, col);
            assert(label >= 0 && label < PALETTE.size());
            cv::Vec3b color = PALETTE[label];
            color_seg.at<cv::Vec3b>(row, col) = color;
        }
    }
    imwrite("/mnt/d/linux_project/linux_project/ort_yolo/rcfile/a.jpg", color_seg);
}
