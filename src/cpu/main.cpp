#include <iostream>
#include <string>
#include <valarray>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "edge_detect.hh"
#include "non_local_means_cpu.hh"
#include "knn.hh"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("usage: main <Image_Path> <Func_name> <Func_Params...>\n");
        return 1;
    }

    string func_name = argv[2];

    if (func_name == "conv" && argc < 4)
    {
        printf("usage: main <Image_Path> conv <Conv_size>\n");
        return 1;
    }
    if (func_name == "knn" && argc < 5)
    {
        printf("usage: main <Image_Path> knn <Conv_size> <Weight_Decay_Param>\n");
        return 1;
    }
    if (func_name == "nlm" && argc < 6)
    {
        printf("usage: main <Image_Path> nlm <Conv_size> <Block_radius> <Weight_Decay_Param>\n");
        return 1;
    }

    // ---- Load image ----
    Mat image;
    if (func_name == "edge_detect")
        image = imread(argv[1], IMREAD_GRAYSCALE);
    else
        image = imread(argv[1], IMREAD_UNCHANGED);

    if (image.empty())
    {
        cerr << "Could not open or find the image" << endl;
        return 1;
    }

    // ---- Process ----
    Mat res;

    if (func_name == "nlm")
        res = non_local_means_cpu(image, stoi(argv[3]), stoi(argv[4]), stod(argv[5]));
    else if (func_name == "knn")
        res = knn(image, stoi(argv[3]), stof(argv[4]));
    else if (func_name == "conv")
        res = convolution(image, stoi(argv[3]));
    else if (func_name == "edge_detect")
    {
        res = knn_grey(image, 2, 150.0);
        res = conv_with_mask(res, 1);
    }
    else
    {
        cerr << "Unknown function name: " << func_name << endl;
        return 1;
    }

    // ---- Display & Save ----
    namedWindow("Display Window", WINDOW_AUTOSIZE);
    imshow("Display Window", res);
    waitKey(0);

    imwrite("output.jpg", res);
    cout << "Output saved as output.jpg" << endl;

    return 0;
}
