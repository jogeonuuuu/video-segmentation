#include "camsub/my_TensorRTEngine.hpp"  // TensorRTEngine 클래스의 선언을 포함하는 헤더 파일
#include "camsub/my_ImageProcessor.hpp"  // 이미지 전처리 및 후처리를 위한 유틸리티 함수들이 정의된 헤더 파일
#include <fstream>  // 파일 입출력을 위한 헤더 파일
#include <iostream> // 표준 입출력 사용을 위한 헤더 파일
#include "cuda_runtime_api.h"  // CUDA 런타임 API 사용을 위한 헤더 파일

// Logger 클래스의 log 함수 구현: 주어진 심각도에 따라 메시지를 출력
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) // 경고 이상의 심각도인 경우 메시지를 출력
        std::cout << msg << std::endl;
}

// TensorRTEngine 클래스 생성자: 멤버 변수 초기화
TensorRTEngine::TensorRTEngine() 
    : runtime(nullptr), engine(nullptr), context(nullptr), inputH(0), inputW(0), numClasses(0) {}

// TensorRTEngine 클래스 소멸자: 자원을 해제하는 cleanup 함수 호출
TensorRTEngine::~TensorRTEngine() {
    cleanup();
}

// 파일을 읽어 벡터에 데이터를 저장하는 함수
void TensorRTEngine::readFile(const std::string& fileName, std::vector<char>& buffer) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate); // 파일을 바이너리 모드로 열고 끝으로 이동
    if (!file.is_open()) { // 파일 열기에 실패한 경우 에러 메시지 출력 후 프로그램 종료
        std::cerr << "파일 열기 실패" << fileName << std::endl;
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg(); // 파일 크기 가져오기
    file.seekg(0, std::ios::beg); // 파일의 시작으로 이동
    buffer.resize(size); // 버퍼의 크기를 파일 크기로 설정
    if (!file.read(buffer.data(), size)) { // 파일 내용을 버퍼에 읽어오기
        std::cerr << "파일 읽기 실패 " << fileName << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ONNX 파일을 사용하여 TensorRT 엔진을 생성하고 파일로 저장하는 함수 원래 함수
/*bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger); // TensorRT 빌더 생성
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0); // 빈 네트워크 정의 생성
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger); // ONNX 파서 생성
    
    std::vector<char> modelData;
    readFile(onnxFile, modelData); // ONNX 모델 파일 읽어오기
    if (!parser->parse(modelData.data(), modelData.size())) { // 모델 파싱
        std::cerr << "onnx모델 파서 실패" << std::endl;
        return false; // 파싱 실패 시 false 반환
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig(); // 빌더 설정 생성
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);  // 작업 공간 메모리 풀 제한 설정 (1GB)

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config); // 직렬화된 엔진 생성
    if (!serializedEngine) { // 엔진 생성 실패 시
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }
    
     // 직렬화된 엔진 데이터를 멤버 변수에 저장
    //serializedEngineData.resize(serializedEngine->size());
    //memcpy(serializedEngineData.data(), serializedEngine->data(), serializedEngine->size());
    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    //엔진 파일 생성끝
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "Failed to create CUDA engine" << std::endl;
        return false;
    }
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0)); // 입력 텐서 차원 가져오기
    inputH = inputDims.d[2]; // 입력 텐서의 높이
    inputW = inputDims.d[3]; // 입력 텐서의 너비
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1)); // 출력 텐서 차원 가져오기
    numClasses = outputDims.d[1]; // 출력 텐서의 클래스 수
    // 할당된 자원 정리
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;
    return true; // 엔진 빌드 성공 시 true 반환
}*/

//변경(1)-빌드 함수
// ONNX 파일을 사용하여 TensorRT 엔진을 생성하고 파일로 저장하거나 기존 엔진을 로드하는 함수  엔진있는지 확인후 없으면생성 
bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    // Step 1: 엔진 파일이 이미 존재하는지 확인
    if (std::filesystem::exists(engineFileName)) {
        // 엔진 파일이 존재하면, 빌드하지 않고 로드
        std::ifstream engineFile(engineFileName, std::ios::binary);
        if (!engineFile.is_open()) {
            std::cerr << "엔진 파일 열기 실패: " << engineFileName << std::endl;
            return false;
        }
        
        // 엔진 파일의 크기를 얻고 데이터를 읽음
        engineFile.seekg(0, std::ios::end);
        std::streamsize engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        
        // 엔진 파일을 메모리에 읽어옴
        std::vector<char> engineData(engineSize);
        if (!engineFile.read(engineData.data(), engineSize)) {
            std::cerr << "엔진 파일 읽기 실패: " << engineFileName << std::endl;
            return false;
        }
        engineFile.close();

        // 엔진 데이터를 이용해 CUDA 엔진을 디시리얼라이즈
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
        if (!engine) {
            std::cerr << "엔진 파일에서 CUDA 엔진 생성 실패" << std::endl;
            return false;
        }

        // 컨텍스트와 입력, 출력 차원 정보 설정
        context = engine->createExecutionContext();
        nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
        nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
        numClasses = outputDims.d[1];

        return true; // 엔진 파일에서 성공적으로 로드
    }

    // Step 2: 엔진 파일이 없을 경우, ONNX 파일로부터 새 엔진을 빌드
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    
    std::vector<char> modelData;
    readFile(onnxFile, modelData);
    if (!parser->parse(modelData.data(), modelData.size())) {
        std::cerr << "ONNX 모델 파서 실패" << std::endl;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }
    
    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "CUDA 엔진 생성 실패" << std::endl;
        return false;
    }

    // 컨텍스트와 입력, 출력 차원 정보 설정
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    numClasses = outputDims.d[1];

    // 자원 해제
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true; // 엔진 빌드 성공 시 true 반환
}

//엔진 생성하여 저장, 기존 엔진이 있는경우 로드 (FP16으로 최적화)
/*bool TensorRTEngine::buildEngine(const std::string& onnxFile, const std::string& engineFileName) {
    // Step 1: 엔진 파일이 이미 존재하는지 확인
    if (std::filesystem::exists(engineFileName)) {
        // 엔진 파일이 존재하면, 빌드하지 않고 로드
        std::ifstream engineFile(engineFileName, std::ios::binary);
        if (!engineFile.is_open()) {
            std::cerr << "엔진 파일 열기 실패: " << engineFileName << std::endl;
            return false;
        }

        // 엔진 파일의 크기를 얻고 데이터를 읽음
        engineFile.seekg(0, std::ios::end);
        std::streamsize engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);

        // 엔진 파일을 메모리에 읽어옴
        std::vector<char> engineData(engineSize);
        if (!engineFile.read(engineData.data(), engineSize)) {
            std::cerr << "엔진 파일 읽기 실패: " << engineFileName << std::endl;
            return false;
        }
        engineFile.close();

        // 엔진 데이터를 이용해 CUDA 엔진을 디시리얼라이즈
        runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
        if (!engine) {
            std::cerr << "엔진 파일에서 CUDA 엔진 생성 실패" << std::endl;
            return false;
        }

        // 컨텍스트와 입력, 출력 차원 정보 설정
        context = engine->createExecutionContext();
        nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
        nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
        numClasses = outputDims.d[1];

        return true; // 엔진 파일에서 성공적으로 로드
    }

    // Step 2: 엔진 파일이 없을 경우, ONNX 파일로부터 새 엔진을 빌드
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    std::vector<char> modelData;
    readFile(onnxFile, modelData);
    if (!parser->parse(modelData.data(), modelData.size())) {
        std::cerr << "ONNX 모델 파서 실패" << std::endl;
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // FP16 모드 활성화
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "엔진 빌드 실패" << std::endl;
        return false;
    }

    std::ofstream engineFile(engineFileName, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
    if (!engine) {
        std::cerr << "CUDA 엔진 생성 실패" << std::endl;
        return false;
    }

    // 컨텍스트와 입력, 출력 차원 정보 설정
    context = engine->createExecutionContext();
    nvinfer1::Dims inputDims = engine->getTensorShape(engine->getIOTensorName(0));
    inputH = inputDims.d[2];
    inputW = inputDims.d[3];
    nvinfer1::Dims outputDims = engine->getTensorShape(engine->getIOTensorName(1));
    numClasses = outputDims.d[1];

    // 자원 해제
    delete serializedEngine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return true; // 엔진 빌드 성공 시 true 반환
}
*/


//변경(2)-추론 함수 - FP32
/*bool TensorRTEngine::runInference(cv::Mat& frame,cv::Mat& result) {
   
    // 컨텍스트가 초기화되지 않은 경우 초기화
    
    // CUDA 버퍼를 한 번만 설정
        int32_t nIO = engine->getNbIOTensors();
        //std::cout<<"nIO::"<<nIO<<std::endl;
        std::vector<std::string> vTensorName(nIO);
        //std::cout<<"vTensorName 개수:"<<vTensorName.size()<<std::endl;
        buffers.resize(nIO);
        bufferSizes.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            vTensorName[i] = std::string(engine->getIOTensorName(i));
            nvinfer1::Dims dims = context->getTensorShape(engine->getIOTensorName(i));
            int64_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            bufferSizes[i] = size * sizeof(float);
            cudaMalloc(&buffers[i], bufferSizes[i]); // 메모리를 한 번만 할당
        }
    // 이미지 전처리 수행
    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(frame, inputBuffer, inputH, inputW)) {
        std::cerr << "Image preprocessing failed." << std::endl;
        delete[] inputBuffer;
        return false;
    }
    //auto start = std::chrono::high_resolution_clock::now();
    // 입력 데이터를 GPU 메모리로 복사
    cudaMemcpy(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice);
    //std::cout<<"이름나오기 직전:"<<std::endl;
    for (int i = 0; i < nIO; ++i) { // defining tensor adress 
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }
    //std::cout<<"오류 부분::"<<vTensorName[2]<<std::endl;
    // 추론 수행
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Inference failed!" << std::endl;
        delete[] inputBuffer;
        cudaStreamDestroy(stream);
        return false;
    }
    cudaStreamSynchronize(stream);
    //auto end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> inferenceTime = end - start;
    //std::cout << "추론 시간: " << inferenceTime.count() << " seconds." << std::endl;
    // 출력 데이터를 CPU 메모리로 복사
    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    cudaMemcpy(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost);
    

    // 세그멘테이션 마스크 생성
    result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);

    // 메모리 해제
    delete[] inputBuffer;
    delete[] outputBuffer;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    cudaStreamDestroy(stream);

    return true; // 결과 마스크 반환
}*/

//변경(3)-추론 함수 - FP32 / 비동기 모드
bool TensorRTEngine::runInference(cv::Mat& frame, cv::Mat& result, cudaStream_t stream) {
   
    // 컨텍스트가 초기화되지 않은 경우 초기화
    
    // CUDA 버퍼를 한 번만 설정
        int32_t nIO = engine->getNbIOTensors();
        //std::cout<<"nIO::"<<nIO<<std::endl;
        std::vector<std::string> vTensorName(nIO);
        //std::cout<<"vTensorName 개수:"<<vTensorName.size()<<std::endl;
        buffers.resize(nIO);
        bufferSizes.resize(nIO);
        for (int i = 0; i < nIO; ++i) {
            vTensorName[i] = std::string(engine->getIOTensorName(i));
            nvinfer1::Dims dims = context->getTensorShape(engine->getIOTensorName(i));
            int64_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            bufferSizes[i] = size * sizeof(float);
            cudaMalloc(&buffers[i], bufferSizes[i]); // 메모리를 한 번만 할당
        }
    // 이미지 전처리 수행
    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(frame, inputBuffer, inputH, inputW)) {
        std::cerr << "Image preprocessing failed." << std::endl;
        delete[] inputBuffer;
        return false;
    }
    //auto start = std::chrono::high_resolution_clock::now();
    // 입력 데이터를 GPU 메모리로 복사
    //cudaMemcpy(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice);
    //std::cout<<"이름나오기 직전:"<<std::endl;
    
    // 입력 데이터를 GPU 메모리로 비동기 복사
    cudaMemcpyAsync(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice, stream);

    for (int i = 0; i < nIO; ++i) { // defining tensor adress 
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }
    //std::cout<<"오류 부분::"<<vTensorName[2]<<std::endl;
    // 추론 수행
    //cudaStream_t stream;
    //cudaStreamCreate(&stream);
    // 비동기 추론 수행
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Inference failed!" << std::endl;
        delete[] inputBuffer;
        return false;
    }

    // 출력 데이터를 비동기로 CPU 메모리로 복사
    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    cudaMemcpyAsync(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost, stream);

    // GPU 작업이 완료될 때까지 기다리도록 동기화 (메인 코드에서)
    // cudaStreamSynchronize(stream); // 이 코드는 메인에서 처리

    

    // 세그멘테이션 마스크 생성
    result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);

    // 메모리 해제
    delete[] inputBuffer;
    delete[] outputBuffer;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }
    //cudaStreamDestroy(stream);

    return true; // 결과 마스크 반환
}

//수정된 추론함수(비동기 모드)GPU CPU 따로 작업 - FP16
/*bool TensorRTEngine::runInference(cv::Mat& frame, cv::Mat& result, cudaStream_t stream){
 int32_t nIO = engine->getNbIOTensors();
    std::vector<std::string> vTensorName(nIO);
    buffers.resize(nIO);
    bufferSizes.resize(nIO);

    for (int i = 0; i < nIO; ++i) {
        vTensorName[i] = std::string(engine->getIOTensorName(i));
        nvinfer1::Dims dims = context->getTensorShape(engine->getIOTensorName(i));
        int64_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        bufferSizes[i] = size * sizeof(float);
        cudaMalloc(&buffers[i], bufferSizes[i]);
    }

    // 이미지 전처리 수행
    float* inputBuffer = new float[bufferSizes[0] / sizeof(float)];
    if (!ImageProcessor::preprocessImage(frame, inputBuffer, inputH, inputW)) {
        std::cerr << "Image preprocessing failed." << std::endl;
        delete[] inputBuffer;
        return false;
    }

    // 입력 데이터를 GPU 메모리로 비동기 복사
    cudaMemcpyAsync(buffers[0], inputBuffer, bufferSizes[0], cudaMemcpyHostToDevice, stream);

    for (int i = 0; i < nIO; ++i) {
        context->setTensorAddress(vTensorName[i].c_str(), buffers[i]);
    }

    // 비동기 추론 수행
    bool status = context->enqueueV3(stream);
    if (!status) {
        std::cerr << "Inference failed!" << std::endl;
        delete[] inputBuffer;
        return false;
    }

    // 출력 데이터를 비동기로 CPU 메모리로 복사
    float* outputBuffer = new float[bufferSizes[1] / sizeof(float)];
    cudaMemcpyAsync(outputBuffer, buffers[1], bufferSizes[1], cudaMemcpyDeviceToHost, stream);

    // GPU 작업이 완료될 때까지 기다리도록 동기화 (메인 코드에서)
    // cudaStreamSynchronize(stream); // 이 코드는 메인에서 처리

    // 세그멘테이션 마스크 생성
    result = ImageProcessor::createSegmentationMask(outputBuffer, inputH, inputW, numClasses);

    // 메모리 해제
    delete[] inputBuffer;
    delete[] outputBuffer;
    for (void* buffer : buffers) {
        cudaFree(buffer);
    }

    return true;
}
*/





// 할당된 자원을 해제하는 함수
void TensorRTEngine::cleanup() {
    if (context) {
        delete context; // 컨텍스트 해제
        context = nullptr;
    }
    if (engine) {
        delete engine; // 엔진 해제
        engine = nullptr;
    }
    if (runtime) {
        delete runtime; // 런타임 해제
        runtime = nullptr;
    }
}
