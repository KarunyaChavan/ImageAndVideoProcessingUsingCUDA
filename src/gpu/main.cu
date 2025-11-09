#include <stdio.h>
#include <iostream>
#include <valarray>
#include <assert.h>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "kernelcall.cuh"

#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define STREL_SIZE 5
#define R (STREL_SIZE / 2)
#define BLOCK_W (TILE_WIDTH + (2 * R))
#define BLOCK_H (TILE_HEIGHT + (2 * R))

static void saveImage(const std::string& path, const cv::Mat& img) {
    if (!cv::imwrite(path, img)) {
        std::cerr << "Failed to write image: " << path << std::endl;
    } else {
        std::cout << "Saved: " << path << std::endl;
    }
}

static bool run_edge_on_gray_frame(const cv::Mat& gray_in, cv::Mat& gray_out)
{
    // gray_in: 1-channel CV_8U
    int width = gray_in.rows;   // follow existing code’s convention
    int height = gray_in.cols;

    Rgb* device_dst_grey = empty_img_device(gray_in);
    double* device_img_grey = img_to_device_grey(gray_in);
    Rgb* out_grey = (Rgb*)malloc(width * height * sizeof(Rgb));

    cv::Mat tmp;
    double otsu_threshold = cv::threshold(gray_in, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    kernel_edge_detect(device_dst_grey, device_img_grey, width, height, 1, otsu_threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(out_grey, device_dst_grey, height * width * sizeof(Rgb), cudaMemcpyDeviceToHost);

    gray_out = gray_in.clone(); // same size/type
    device_to_img_grey(out_grey, gray_out);

    cudaFree(device_dst_grey);
    cudaFree(device_img_grey);
    free(out_grey);
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "usage: main <Path> <Func_name>\n"
                     "Funcs (images): pixelize, conv, shared_conv, knn, shared_knn, nlm, edge_detect\n"
                     "Funcs (video): edge_detect_video\n"
                  << std::endl;
        return 1;
    }
    std::string func_name = argv[2];

    // Video mode: edge_detect_video
    if (func_name == "edge_detect_video")
    {
        const std::string in_path = argv[1];
        cv::VideoCapture cap(in_path);
        if (!cap.isOpened()) {
            std::cerr << "Could not open video: " << in_path << std::endl;
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0; // fallback
        int frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
        double seconds = (fps > 0 && frame_count > 0) ? (frame_count / fps) : -1;

        if (seconds > 10.0) {
            std::cerr << "[Warn] Input video is ~" << seconds
                      << "s (>10s). It will still be processed, but expect longer runtime."
                      << std::endl;
        }

        cv::VideoWriter writer;
        // Use MP4 container with MPEG-4 codec ('mp4v') to be widely compatible on Colab
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        if (!writer.open("output_edges.mp4", fourcc, fps, cv::Size(frame_w, frame_h), true)) {
            std::cerr << "Failed to open VideoWriter for output_edges.mp4" << std::endl;
            return 1;
        }

        cv::Mat frame, gray, gray_edges, bgr_out;
        size_t idx = 0;
        while (cap.read(frame))
        {
            if (frame.empty()) break;

            // Ensure 3-channel BGR input
            if (frame.channels() == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
            }

            // Grayscale
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // Run CUDA edge pipeline on this frame
            if (!run_edge_on_gray_frame(gray, gray_edges)) {
                std::cerr << "Edge detection failed on frame " << idx << std::endl;
                return 1;
            }

            // Writer expects 3-channel by default: convert gray->BGR
            cv::cvtColor(gray_edges, bgr_out, cv::COLOR_GRAY2BGR);
            writer.write(bgr_out);

            ++idx;
        }

        writer.release();
        cap.release();
        std::cout << "Saved output video: output_edges.mp4" << std::endl;
        return 0;
    }

    // Image modes below (original behavior, headless save)
    if (func_name == "knn" && argc < 5)
    {
        std::cout << "usage: main <Image_Path> knn <Conv_size> <Weight_Decay_Param>" << std::endl;
        return 1;
    }
    if (func_name == "nlm" && argc < 6)
    {
        std::cout << "usage: main <Image_Path> nlm <Conv_size> <Block_radius> <Weight_Decay_Param>" << std::endl;
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);
    if (!image.data)
    {
        std::cout << "Could not open or find the image" << std::endl;
        return 1;
    }

    int width = image.rows;
    int height = image.cols;

    Rgb* device_dst;
    Rgb* device_img;
    Rgb* out;
    Rgb* device_dst_grey;
    double* device_img_grey;
    Rgb* out_grey;

    cv::Mat grey_img;
    device_dst = empty_img_device(image);
    device_img = img_to_device(image);
    out = (Rgb*)malloc(width * height * sizeof (Rgb));

    if (func_name == "pixelize")
        kernel_pixelize_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "conv")
        kernel_conv_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "shared_conv")
        kernel_shared_conv_host(device_dst, device_img, width, height, std::stoi(argv[3]));
    else if (func_name == "knn")
        kernel_knn_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stod(argv[4]));
    else if (func_name == "shared_knn")
        kernel_shared_knn_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stod(argv[4]));
    else if (func_name == "nlm")
        kernel_nlm_host(device_dst, device_img, width, height, std::stoi(argv[3]), std::stoi(argv[4]), std::stod(argv[5]));
    else if (func_name == "edge_detect")
    {
        cv::cvtColor(image, grey_img, cv::COLOR_BGR2GRAY);
        device_dst_grey = empty_img_device(grey_img);
        device_img_grey = img_to_device_grey(grey_img);
        out_grey = (Rgb*)malloc(width * height * sizeof (Rgb));
        cv::Mat dst;
        double otsu_threshold = cv::threshold(grey_img, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        kernel_edge_detect(device_dst_grey, device_img_grey, width, height, 1, otsu_threshold);
    }
    else
    {
        std::cout << "error: function name '" << func_name << "' is not known." << std::endl;
        cudaFree(device_dst);
        cudaFree(device_img);
        free(out);
        return 1;
    }

    cudaDeviceSynchronize();

    if (func_name != "edge_detect")
    {
        cudaMemcpy(out, device_dst, height * width * sizeof (Rgb), cudaMemcpyDeviceToHost);
        device_to_img(out, image);
        cudaFree(device_dst);
        cudaFree(device_img);
        free(out);
        saveImage("output.png", image);
    }
    else
    {
        cudaMemcpy(out_grey, device_dst_grey, height * width * sizeof (Rgb), cudaMemcpyDeviceToHost);
        device_to_img_grey(out_grey, grey_img);
        cudaFree(device_dst_grey);
        cudaFree(device_img_grey);
        free(out_grey);
        saveImage("output_edges.png", grey_img);
    }
    grey_img.release();
    image.release();
    return 0;
}