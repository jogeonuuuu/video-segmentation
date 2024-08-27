

#include "camsub/my_ImageProcessor.hpp"  
#include <iostream>               
#include <algorithm>              
#include <cmath>                  
#include <vector>                 

// 이미지 전처리 함수: 이미지 파일을 읽고 모델의 입력에 맞게 전처리
/*bool ImageProcessor::preprocessImage(const std::string& imageFile, float* buffer, int inputH, int inputW) {
    cv::Mat img = cv::imread(imageFile);  // 이미지 파일읽기
    if (img.empty()) {  // 이미지 파일을 제대로 읽지 못했을 경우
        std::cerr << "이미지 로드 실패" << imageFile << std::endl;  // 오류 메시지 출력
        return false;  
    }

    cv::Mat resized;  // 리사이즈된 이미지를 저장할 행렬 선언
    cv::resize(img, resized, cv::Size(inputW, inputH));  // 이미지를 지정된 크기로 리사이즈
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);  // 이미지의 픽셀 값을 0-1 범위로 정규화 (float 형식)

    cv::imshow("src", resized);  // 리사이즈된 이미지를 화면에 표시 (디버깅 용도)
    cv::waitKey(1);  

    std::vector<cv::Mat> channels(3);  // BGR 채널을 저장할 벡터 생성
    cv::split(resized, channels);  // 리사이즈된 이미지를 B, G, R 세 개의 채널로 분리

    for (int i = 0; i < 3; ++i) {  // 각 채널에 대해 반복
        std::memcpy(buffer + i * inputH * inputW, channels[i].data, inputH * inputW * sizeof(float));  // 각 채널의 데이터를 모델 입력 버퍼에 복사
    }
    return true;  // 성공을 나타내는 true 반환
}*/

// 소프트맥스 함수: 입력 배열을 확률로 변환하는 함수
void ImageProcessor::softmax(float* input, int size) {
    float max = *std::max_element(input, input + size);  // 입력 배열에서 최대값을 찾아서 변수 max에 저장
    float sum = 0;  // 소프트맥스 계산을 위한 합 변수 초기화
    for (int i = 0; i < size; i++) {  // 입력 배열의 각 원소에 대해 반복
        input[i] = std::exp(input[i] - max);  // 입력 값에서 최대값을 뺀 후 지수 함수 적용
        sum += input[i];  // 지수 함수 결과를 sum에 추가하여 합 계산
    }
    for (int i = 0; i < size; i++) {  // 다시 입력 배열의 각 원소에 대해 반복
        input[i] /= sum;  // 각 원소를 합으로 나누어 확률로 변환
    }
}

// 세그멘테이션 마스크 생성 함수: 모델의 출력을 기반으로 세그멘테이션 마스크를 생성
/*void ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses) {
    int outputH = inputH;  // 출력 이미지의 높이를 입력 이미지의 높이로 설정
    int outputW = inputW;  // 출력 이미지의 너비를 입력 이미지의 너비로 설정

    cv::Mat segmentationMask(outputH, outputW, CV_8UC1);  // 세그멘테이션 마스크를 저장할 그레이스케일 이미지 (8비트, 단일 채널) 생성
    std::vector<int> classCounts(numClasses, 0);  // 각 클래스의 픽셀 개수를 저장할 벡터 초기화
    for (int y = 0; y < outputH; ++y) {  // 이미지의 각 픽셀 행에 대해 반복
        for (int x = 0; x < outputW; ++x) {  // 이미지의 각 픽셀 열에 대해 반복
            std::vector<float> pixelProbs(numClasses);  // 각 픽셀에 대해 클래스 확률을 저장할 벡터 생성
            for (int c = 0; c < numClasses; ++c) {  // 각 클래스에 대해 반복
                pixelProbs[c] = outputBuffer[(c * outputH * outputW) + (y * outputW) + x];  // 모델의 출력 버퍼에서 해당 픽셀의 클래스 확률을 가져와 저장
            }
            
            softmax(pixelProbs.data(), numClasses);  // 픽셀별 클래스 확률에 소프트맥스 적용하여 확률 값으로 변환

            int maxClass = std::max_element(pixelProbs.begin(), pixelProbs.end()) - pixelProbs.begin();  // 가장 높은 확률을 가진 클래스 인덱스 찾기
            segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));  // 클래스 인덱스를 0-255 사이로 변환하여 마스크에 저장
            classCounts[maxClass]++;  // 해당 클래스의 픽셀 개수 증가
        }
    }

    cv::Mat resizedMask;  // 리사이즈된 마스크 이미지를 저장할 행렬 선언
    cv::resize(segmentationMask, resizedMask, cv::Size(inputW, inputH), 0, 0, cv::INTER_NEAREST);  // 세그멘테이션 마스크를 원본 크기로 리사이즈
    cv::Mat colorMask;  // 컬러맵이 적용된 마스크 이미지를 저장할 행렬 선언
    cv::applyColorMap(resizedMask, colorMask, cv::COLORMAP_JET);  // 컬러맵을 적용하여 시각화하기 쉽게 변환
    cv::imshow("colorMask", colorMask);  // 컬러 마스크 이미지를 화면에 표시
    cv::waitKey(0);  // 키 입력 대기 (무한 대기)
    //cv::imwrite("segmentation_mask.png", colorMask);  // 컬러 마스크 이미지를 파일로 저장
    //std::cout << "Segmentation mask saved as 'segmentation_mask.png'" << std::endl;
    //cv::imwrite("segmentation_mask_gray.png", resizedMask);  // 그레이스케일 마스크 이미지를 파일로 저장
    //std::cout << "Grayscale segmentation mask saved as 'segmentation_mask_gray.png'" << std::endl;
}*/
// 새로 추가된 cv::Mat 객체를 인자로 받는 함수 정의
bool ImageProcessor::preprocessImage(const cv::Mat& image, float* buffer, int inputH, int inputW) {
    if (image.empty()) {
        std::cerr << "Image is empty." << std::endl;
        return false;
    }
    //cv::imshow("src",image);
    //cv::waitKey(0);
    // 입력 크기가 0이 아닌지 확인
    if (inputW <= 0 || inputH <= 0) {
        std::cerr << "Invalid input dimensions: " << inputW << "x" << inputH << std::endl;
        return false;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(inputW, inputH)); // 이미지 크기 조정
    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f); // 0-255 범위를 0-1로 정규화
    //cv::imshow("src",resized);
    //cv::waitKey(0);
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int i = 0; i < 3; ++i) {
        std::memcpy(buffer + i * inputH * inputW, channels[i].data, inputH * inputW * sizeof(float));
    }
    //cv::imshow("src",resized);
    //cv::waitKey(0);
    return true;
}

cv::Mat ImageProcessor::createSegmentationMask(float* outputBuffer, int inputH, int inputW, int numClasses) {
    cv::Mat segmentationMask(inputH, inputW, CV_8UC1);
    for (int y = 0; y < inputH; ++y) {
        for (int x = 0; x < inputW; ++x) {
            int maxClass = 0;
            float maxProb = outputBuffer[y * inputW + x];
            for (int c = 1; c < numClasses; ++c) {
                float prob = outputBuffer[(c * inputH * inputW) + (y * inputW) + x];
                if (prob > maxProb) {
                    maxClass = c;
                    maxProb = prob;
                }
            }
            segmentationMask.at<uchar>(y, x) = static_cast<uchar>(maxClass * 255 / (numClasses - 1));
        }
    }
    cv::Mat colorMask;
    cv::applyColorMap(segmentationMask, colorMask, cv::COLORMAP_JET);
    return colorMask;
}