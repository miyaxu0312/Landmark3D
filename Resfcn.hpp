//
//  Resfcn.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef Resfcn_hpp
#define Resfcn_hpp

#include "Net.hpp"
#include "util.hpp"
#include <stdio.h>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"
#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
namespace Shadow
{
    class Resfcn: public Net
    {
    public:
        Resfcn(int batchSize, const int *inputShape, float *preParam, InterMethod interMethod);
        ShadowStatus init(const int gpuID, void *data, const int batchSize);
        ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results);
        ShadowStatus destroy();
        vector<vector<float>>  get_landmark_result();
    private:
        ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize, IHostMemory*& trtModelStream);
        ShadowStatus doInference(float* inputData, float* outputData, int batchSize);
        string locateFile(const std::string& input);
        void* safeCudaMalloc(size_t memSize);
        vector<std::pair<int64_t, nvinfer1::DataType>>
        calculateBindingBufferSizes(int nbBindings, int batchSize);
        
        vector<float> pre_process(const vector<cv::Mat> &imgs,string attribute, vector<Affine_Matrix> &affine_matrix);
        void post_process(vector<Affine_Matrix> &affine_matrix, vector<Mat> &network_out, vector<Mat> &position_map);
        void dealResult(vector<std::string> &results, vector<Mat> &position_map);
        void getResultJson(vector<vector<float>> &landmark_one, vector<float> &pose, vector<std::string> &results);
        
        ICudaEngine *engine;
        IExecutionContext *context;
        IUffParser* parser;
        //初始化参数列表
        int BATCH_SIZE;
        int INPUT_CHANNELS;
        int INPUT_WIDTH;
        int INPUT_HEIGHT;
        
        int inputIndex;
        int outputIndex;
        
        int img_num = 1;
        int run_num = 1;
        void *buffers[2];
        int iteration = 1;
        int resolution = 256;
        string suffix = ".*.jpg";
        
        
        const char* INPUT_BLOB_NAME = "Placeholder";
        const char* OUTPUT_BLOB_NAME = = "resfcn256/Conv2d_transpose_16/Sigmoid";
        const char* UFF_MODEL_PATH = "/workspace/run/xyx/TensorRT-4.0.1.6/data/landmark/face.pb.uff";
        const string json_result_path = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/Landmark3D/landmark.json";
        
        vector<vector<float>>  landmark_result;
        
        
};
}
#endif /* Resfcn_hpp */
