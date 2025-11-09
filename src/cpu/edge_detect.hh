#pragma once
#include <iostream>
#include <valarray>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat convolution(const cv::Mat& image, int conv_size);
cv::Mat conv_with_mask(const cv::Mat& image, int conv_size);
std::valarray<double> simple_conv(const cv::Mat& image, int x, int y, int conv_size);
