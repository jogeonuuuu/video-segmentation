#pragma once

#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <filesystem> // C++17에서 std::filesystem 사용을 위한 헤더 
#include <fstream>
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};
class TensorRTEngine {
public:
    TensorRTEngine();
    ~TensorRTEngine();
    bool buildEngine(const std::string& onnxFile, const std::string& engineFileName);// 엔진을 생성하는 멤버함수
    //bool runInference(const std::string& imageFile);// 생성된 엔진으로 추론을 수행하는 멤버함수 -> (image)
    //bool runInference(cv::Mat& frame,cv::Mat& result);//동영상으로 추론을 진행하는 멤버함수 -> (video)
    bool runInference(cv::Mat& frame, cv::Mat& result, cudaStream_t stream);//비동기 추론 모드함수
    bool processVideo(const std::string& videoFile);//동영상 불러오기
private:
    Logger logger;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;
    std::vector<void*> buffers;
    std::vector<int64_t> bufferSizes;
    int inputH, inputW, numClasses;
    //std::vector<char> serializedEngineData; // 직렬화된 엔진 데이터를 저장할 멤버 변수 추가 runInfer 멤버함수에서 사용하기위해 멤버변수로 선언해 저장
    void readFile(const std::string& fileName, std::vector<char>& buffer);
    void cleanup();
};
