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
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
namespace Landmark
{
    class Resfcn: public Net
    {
    public:
        Resfcn(const int batchSize, const int *inputShape);
        LandmarkStatus init(const int gpuID, const int batchSize, const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, const int MaxBatchSize, int n);
        LandmarkStatus predict(vector<IMAGE> &imgs, const string boxPath, const string uv_kpt_ind, const string faceIndex, const string canonical_vertices_path, int resolution, const string suffix, const int iteration, string json_result_path);
        LandmarkStatus destroy();
        
    private:
        ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize, IHostMemory*& trtModelStream);
        LandmarkStatus doInference(float* inputData, float* outputData, int batchSize);
        string locateFile(const std::string& input);
        void* safeCudaMalloc(size_t memSize);
        vector<std::pair<int64_t, nvinfer1::DataType>>
        calculateBindingBufferSizes(int nbBindings, int batchSize);
        
        vector<float> pre_process(const string boxPath, int resolution, const string suffix);
        void post_process(const string canonical_vertices_path, const string faceIndex, const string uv_kpt_ind_path, int resolution, const string suffix, string json_result_path);
        void dealResult(const int resolution, const string faceIndex, const string uv_kpt_ind_path,  string json_result_path, string canonical_vertices_path);
        void getResultJson(vector<vector<float>> &landmark_one, vector<float> &pose, string name, string json_result_path);
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
        
        int img_num;
        int run_num;
        void *buffers[2];
        int iteration;
        
        const char* INPUT_BLOB_NAME;
        const char* OUTPUT_BLOB_NAME;
        
        vector<IMAGE> ori_img;
        vector<IMAGE> network_out;
        vector<IMAGE> position_map;
        vector<Affine_Matrix> affine_matrix;
        vector<LANDMARK> landmark;
};
}
#endif /* Resfcn_hpp */
