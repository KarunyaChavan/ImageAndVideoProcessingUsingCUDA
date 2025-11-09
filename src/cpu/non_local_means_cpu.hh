#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat non_local_means_cpu(const cv::Mat& image, int search_radius, int patch_radius, double h_param);
