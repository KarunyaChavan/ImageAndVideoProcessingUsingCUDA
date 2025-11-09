#include <cmath>
#include "edge_detect.hh"

using namespace cv;
using namespace std;

// =====================================================
// Helper: simple mean convolution (already discussed)
// =====================================================
std::valarray<double> simple_conv(const cv::Mat& image, int x, int y, int conv_size)
{
    std::valarray<double> rgb = {0.0, 0.0, 0.0};
    int count = 0;

    for (int j = y - conv_size; j <= y + conv_size; ++j)
    {
        for (int i = x - conv_size; i <= x + conv_size; ++i)
        {
            if (i >= 0 && j >= 0 && i < image.cols && j < image.rows)
            {
                auto pix = image.at<cv::Vec3b>(j, i);
                rgb[0] += pix[0];
                rgb[1] += pix[1];
                rgb[2] += pix[2];
                ++count;
            }
        }
    }
    if (count > 0) rgb /= count;
    return rgb;
}

// =====================================================
// Simple convolution across the entire image
// =====================================================
cv::Mat convolution(const cv::Mat& image, int conv_size)
{
    cv::Mat conv_img = image.clone();

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            auto gc = simple_conv(image, x, y, conv_size);
            conv_img.at<cv::Vec3b>(y, x) = cv::Vec3b(gc[0], gc[1], gc[2]);
        }
    }
    return conv_img;
}

// =====================================================
// Compute gradient magnitude & direction using masks
// =====================================================
static std::pair<double, double> conv_mask_pixel(
    const cv::Mat& image, int x, int y,
    const int maskX[3][3], const int maskY[3][3])
{
    double gx = 0.0, gy = 0.0;

    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            int xx = x + i, yy = y + j;
            if (xx >= 0 && yy >= 0 && xx < image.cols && yy < image.rows)
            {
                double val = image.at<uchar>(yy, xx);
                gx += val * maskX[j + 1][i + 1];
                gy += val * maskY[j + 1][i + 1];
            }
        }
    }

    double magnitude = std::sqrt(gx * gx + gy * gy);
    double angle = std::atan2(gy, gx) * 180.0 / M_PI;
    if (angle < 0) angle += 180.0;

    return { magnitude, angle };
}

// =====================================================
// Non-Maximum Suppression
// =====================================================
static cv::Mat non_max_suppression(const cv::Mat& grad, const cv::Mat& dir, double threshold)
{
    cv::Mat suppressed = cv::Mat::zeros(grad.size(), CV_8UC1);

    for (int y = 1; y < grad.rows - 1; ++y)
    {
        for (int x = 1; x < grad.cols - 1; ++x)
        {
            double angle = dir.at<float>(y, x);
            double mag = grad.at<float>(y, x);
            if (mag < threshold)
                continue;

            double q = 0, r = 0;

            // Quantize gradient direction
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180))
            {
                q = grad.at<float>(y, x + 1);
                r = grad.at<float>(y, x - 1);
            }
            else if (22.5 <= angle && angle < 67.5)
            {
                q = grad.at<float>(y + 1, x - 1);
                r = grad.at<float>(y - 1, x + 1);
            }
            else if (67.5 <= angle && angle < 112.5)
            {
                q = grad.at<float>(y + 1, x);
                r = grad.at<float>(y - 1, x);
            }
            else if (112.5 <= angle && angle < 157.5)
            {
                q = grad.at<float>(y - 1, x - 1);
                r = grad.at<float>(y + 1, x + 1);
            }

            if (mag >= q && mag >= r)
                suppressed.at<uchar>(y, x) = 255;
        }
    }
    return suppressed;
}

// =====================================================
// Hysteresis thresholding — connect weak edges
// =====================================================
static void hysteresis(cv::Mat& edges, double low_thresh, double high_thresh)
{
    cv::Mat strong_edges;
    cv::threshold(edges, strong_edges, high_thresh, 255, cv::THRESH_BINARY);

    cv::Mat weak_edges;
    cv::threshold(edges, weak_edges, low_thresh, 255, cv::THRESH_BINARY);
    weak_edges -= strong_edges;

    // Connect weak edges to strong ones
    for (int y = 1; y < edges.rows - 1; ++y)
    {
        for (int x = 1; x < edges.cols - 1; ++x)
        {
            if (weak_edges.at<uchar>(y, x) == 255)
            {
                bool connected = false;
                for (int j = -1; j <= 1; ++j)
                {
                    for (int i = -1; i <= 1; ++i)
                    {
                        if (strong_edges.at<uchar>(y + j, x + i) == 255)
                        {
                            connected = true;
                            break;
                        }
                    }
                    if (connected) break;
                }

                if (connected)
                    edges.at<uchar>(y, x) = 255;
                else
                    edges.at<uchar>(y, x) = 0;
            }
        }
    }
}

// =====================================================
// Main Canny-like function
// =====================================================
cv::Mat conv_with_mask(const cv::Mat& image, int conv_size)
{
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    gray.convertTo(gray, CV_8UC1);
    int maskX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int maskY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    cv::Mat grad(gray.size(), CV_32FC1);
    cv::Mat dir(gray.size(), CV_32FC1);

    for (int y = 0; y < gray.rows; ++y)
    {
        for (int x = 0; x < gray.cols; ++x)
        {
            auto [magnitude, angle] = conv_mask_pixel(gray, x, y, maskX, maskY);
            grad.at<float>(y, x) = static_cast<float>(magnitude);
            dir.at<float>(y, x) = static_cast<float>(angle);
        }
    }

    cv::Mat dummy;
    double otsu_thresh = cv::threshold(gray, dummy, 0, 255,
                                       cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat grad_norm;
    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat edges = non_max_suppression(grad, dir, otsu_thresh);
    hysteresis(edges, otsu_thresh * 0.5, otsu_thresh);

    return edges;
}
