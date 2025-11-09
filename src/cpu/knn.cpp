#include <cmath>
#include "knn.hh"

// Gaussian-weighted bilateral-like convolution
std::valarray<double> gauss_conv(const cv::Mat& image, int x, int y, int conv_size, double h_param)
{
    std::valarray<double> rgb = {0.0, 0.0, 0.0};
    std::valarray<double> cnt = {0.0, 0.0, 0.0};

    for (int j = y - conv_size; j <= y + conv_size; ++j)
    {
        for (int i = x - conv_size; i <= x + conv_size; ++i)
        {
            if (i >= 0 && j >= 0 && i < image.cols && j < image.rows)
            {
                auto ux = image.at<cv::Vec3b>(y, x);
                auto uy = image.at<cv::Vec3b>(j, i);
                double c1 = std::exp(-((i - x)*(i - x) + (j - y)*(j - y)) / (2.0 * std::pow(conv_size, 2)));
                double h2 = std::pow(h_param, 2);

                for (int k = 0; k < 3; ++k)
                {
                    double c2 = std::exp(-std::pow(uy[k] - ux[k], 2) / h2);
                    double w = c1 * c2;
                    rgb[k] += uy[k] * w;
                    cnt[k] += w;
                }
            }
        }
    }

    for (int k = 0; k < 3; ++k)
        if (cnt[k] != 0) rgb[k] /= cnt[k];

    return rgb;
}

cv::Mat knn(const cv::Mat& image, int conv_size, double h_param)
{
    cv::Mat knn_img = image.clone();

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            auto gc = gauss_conv(image, x, y, conv_size, h_param);
            knn_img.at<cv::Vec3b>(y, x) = cv::Vec3b(gc[0], gc[1], gc[2]);
        }
    }
    return knn_img;
}

<<<<<<< HEAD
cv::Mat knn_grey(const cv::Mat& image, int conv_size, double h_param)
{
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::Mat dst = gray.clone();
    double h2 = std::pow(h_param, 2);

    for (int y = 0; y < gray.rows; ++y)
    {
        for (int x = 0; x < gray.cols; ++x)
        {
            double num = 0.0;
            double den = 0.0;
            double center = gray.at<uchar>(y, x);

            for (int j = y - conv_size; j <= y + conv_size; ++j)
            {
                for (int i = x - conv_size; i <= x + conv_size; ++i)
                {
                    if (i >= 0 && j >= 0 && i < gray.cols && j < gray.rows)
                    {
                        double neighbor = gray.at<uchar>(j, i);

                        double c1 = std::exp(-((i - x)*(i - x) + (j - y)*(j - y)) / (2.0 * std::pow(conv_size, 2)));
                        double c2 = std::exp(-std::pow(neighbor - center, 2) / h2);
                        double w = c1 * c2;

                        num += neighbor * w;
                        den += w;
                    }
                }
            }

            if (den != 0)
                dst.at<uchar>(y, x) = static_cast<uchar>(num / den);
        }
    }

    return dst;
}

=======

// Applies a gauss convolution given a location and a grey image
double gauss_conv_gray(cv::Mat image, int x, int y, int conv_size, double h_param)
{
    double rgb = 0;
    double cnt = 0;
    int cx = 0;
    for (int j = y - conv_size; j < y + conv_size; j++)
    {
        for (int i = x - conv_size; i < x + conv_size; i++)
        {
            if (i >= 0 and j >= 0)
            {
                auto ux = image.at<uchar>(x, y);
                auto uy = image.at<uchar>(i, j);
                double c1 = std::exp(-(std::pow(std::abs(i + j - (x + y)), 2)) / (float)std::pow(conv_size, 2));
                double h_div = std::pow(h_param, 2);

                double c2 = std::exp(-(std::pow(std::abs(uy - ux), 2)) / h_div);

                rgb += uy * c1 * c2;

                cnt += c1 * c2;
            }
        }
    }
    if (cnt != 0)
    {
        rgb /= cnt;
    }
    return rgb;
}


// Hat function of the K-Nearest Neighbors algorithm on a grey image
// Removes the noise of a grey image
cv::Mat knn_grey(cv::Mat image, int conv_size, double h_param)
{
    auto knn_img = image.clone();
    for (int y = 0; y < image.cols; y++)
    {
        for (int x = 0; x < image.rows; x++)
        {
           auto gc = gauss_conv_gray(image, x, y, conv_size, h_param);
           knn_img.at<uchar>(x, y) = gc;
        }
    }
    return knn_img;
}
>>>>>>> upstream/master
