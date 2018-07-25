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
using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;
namespace Landmark
{
    class Resfcn: public Net
    {
    public:
        Resfcn(const int batchSize, const int *inputShape);
        LandmarkStatus init(const int gpuID, const int batchSize, const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, const int MaxBatchSize);
        LandmarkStatus predict(const string img_path, const string result_path, const string suffix, const int iteration, vector<Affine_Matrix> &affine_matrix);
        LandmarkStatus destroy();
    private:
        ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize, IHostMemory*& trtModelStream);
        LandmarkStatus doInference(float* inputData, float* outputData, int batchSize);
        string locateFile(const std::string& input);
        void* safeCudaMalloc(size_t memSize);
        vector<std::pair<int64_t, nvinfer1::DataType>>
        calculateBindingBufferSizes(int nbBindings, int batchSize);
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
        map<int,string> img_name;
    };
}
#endif /* Resfcn_hpp */
