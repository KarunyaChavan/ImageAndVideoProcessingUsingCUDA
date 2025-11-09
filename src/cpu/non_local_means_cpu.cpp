#include <cmath>
#include <valarray>   // ✅ Added this include
#include "non_local_means_cpu.hh"

using namespace cv;
using namespace std;

// Compute squared patch distance between two pixel neighborhoods
static std::valarray<double> patch_distance(const Mat& image, int x1, int y1, int x2, int y2, int patch_radius)
{
    std::valarray<double> diff = {0.0, 0.0, 0.0};
    int count = 0;

    for (int j = -patch_radius; j <= patch_radius; ++j)
    {
        for (int i = -patch_radius; i <= patch_radius; ++i)
        {
            int xx1 = x1 + i, yy1 = y1 + j;
            int xx2 = x2 + i, yy2 = y2 + j;

            if (xx1 >= 0 && yy1 >= 0 && xx2 >= 0 && yy2 >= 0 &&
                xx1 < image.cols && yy1 < image.rows &&
                xx2 < image.cols && yy2 < image.rows)
            {
                auto p1 = image.at<Vec3b>(yy1, xx1);
                auto p2 = image.at<Vec3b>(yy2, xx2);

                for (int k = 0; k < 3; ++k)
                    diff[k] += std::pow(p1[k] - p2[k], 2);

                ++count;
            }
        }
    }

    if (count > 0)
        diff /= count;

    return diff;
}

// Compute NLM-weighted pixel value
static std::valarray<double> nlm_pixel(const Mat& image, int x, int y, int search_radius, int patch_radius, double h_param)
{
    std::valarray<double> num = {0.0, 0.0, 0.0};
    std::valarray<double> den = {0.0, 0.0, 0.0};
    double h2 = std::pow(h_param, 2);

    for (int j = -search_radius; j <= search_radius; ++j)
    {
        for (int i = -search_radius; i <= search_radius; ++i)
        {
            int nx = x + i, ny = y + j;
            if (nx >= 0 && ny >= 0 && nx < image.cols && ny < image.rows)
            {
                auto d = patch_distance(image, x, y, nx, ny, patch_radius);
                auto p = image.at<Vec3b>(ny, nx);

                for (int k = 0; k < 3; ++k)
                {
                    double w = std::exp(-d[k] / h2);
                    num[k] += w * p[k];
                    den[k] += w;
                }
            }
        }
    }

    for (int k = 0; k < 3; ++k)
        if (den[k] != 0)
            num[k] /= den[k];

    return num;
}

// Main Non-local Means function
cv::Mat non_local_means_cpu(const Mat& image, int search_radius, int patch_radius, double h_param)
{
    Mat dst = image.clone();

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            auto gc = nlm_pixel(image, x, y, search_radius, patch_radius, h_param);
            dst.at<Vec3b>(y, x) = Vec3b(gc[0], gc[1], gc[2]);
        }
    }

    return dst;
}
