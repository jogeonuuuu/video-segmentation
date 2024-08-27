#pragma once

#include <string>
#include <opencv2/opencv.hpp>

class ImageProcessor {
public:
    // 이미지 파일 경로를 인자로 받는 기존 함수
    //static bool preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW);
    //static void createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
    // cv::Mat 객체를 인자로 받는 새 함수
    static bool preprocessImage(const cv::Mat& image, float* buffer, int inputH, int inputW);

    // cv::Mat 타입의 마스크를 반환하도록 수정된 함수
    static cv::Mat createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses);
private:
    // 후처리 함수에서 사용할 소프트맥스 함수
    static void softmax(float* input, int size);
};
