//
//  Resfcn.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Resfcn.hpp"
#include "util.hpp"
#include <vector>
#include <dirent.h>
#include <io.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;

namespace Landmark {
static Logger gLogger;
static samples_common::Args args;
#define MAX_WORKSPACE (1<<30)
    
    Resfcn::Resfcn(const int batchSize, const int *inputShape): 
    {
        this->BATCH_SIZE = batchSize;
	this->INPUT_CHANNELS = inputShape[0];
	this->INPUT_WIDTH = inputShape[1];
	this->INPUT_HEIGHT = inputShape[2];
    }
    
    LandmarkStatus Resfcn::init(const int gpuID, const int batchSize, const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, const int MaxBatchSize)
    {
        this->INPUT_BLOB_NAME = INPUT_BLOB_NAME;
        this->OUTPUT_BLOB_NAME = OUTPUT_BLOB_NAME;
        
        string fileName = UFF_MODEL_PATH;
        parser = createUffParser();
        
        try
        {
            parser->registerInput(INPUT_BLOB_NAME, Dims3(INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT), UffInputOrder::kNCHW);
            parser->registerOutput(OUTPUT_BLOB_NAME);
        }catch(...)
        {
            return landmark_status_blobname_error;
        }
        
        IHostMemory* trtModelStream{nullptr};
        
        ICudaEngine* tmpengine = loadModelAndCreateEngine(fileName.c_str(), MaxBatchSize,trtModelStream);
	assert(trtModelStream != nullptr);
	if (!tmpengine)
	    return landmark_status_create_model_error;
	tmpengine->destroy();
	    
	try
    	{
            cudaSetDevice(gpuID);
    	}catch (...)
    	{
            return landmark_status_set_gpu_error;
    	}

        IRuntime* runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);
        try
        {
            engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(),nullptr);
        }catch(...)
        {
            return landmark_status_deserialize_error;
        }
	    
        assert(engine != nullptr);
        trtModelStream->destroy();
        context = engine->createExecutionContext();
        assert(context != nullptr);
	    
        return landmark_status_success;
    }
    
    LandmarkStatus Resfcn::predict(const string img_path, const string result_path, const string suffix, const int iteration, vector<Affine_Matrix> &affine_matrix)
    {
        vector<string> files;
        vector<string> split_result;
		Mat img;
        if (access(img_path.c_str(),6) == -1)
        {
            return landmark_status_data_error;
        }
        files = get_all_files(img_path, suffix);
        img_num = files.size();
        this->run_num = img_num;
        this->iteration = iteration;
        
        vector<float> networkOut(img_num * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
        vector<float> data;
        
        int num = 0;
        string tmpname;
        cout<<"prepare data..."<<endl;
        for(int i = 0; i < img_num; ++i)
        {
            bool isfind = false;
            split_result = my_split(files[i],"/");
            tmpname = split_result[split_result.size()-1];
            img_name.insert(pair<int, string>(i, tmpname));
            vector<Affine_Matrix>::iterator iter;
            for(iter = affine_matrix.begin(); iter!= affine_matrix.end(); ++iter)
            {
                if((*iter).name == tmpname)
                {
                    img = (*iter).crop_img;
                    isfind = true;
                    continue;
                }
            }
            img.convertTo(img, CV_32FC3);
            if( !isfind )
                continue;
            for(int c=0; c<INPUT_CHANNELS; ++c)
            {
                for(int row=0; row<INPUT_WIDTH; row++)
                {
                    for(int col=0; col<INPUT_HEIGHT; col++, ++num)
                    {
                        data.push_back(img.at<Vec3f>(row,col)[c]);
                    }
                }
            }
        }
        LandmarkStatus status = doInference(&data[0], &networkOut[0], BATCH_SIZE);
        std::cout<<"Inference uploaded..."<<endl;
        
        float* outdata=nullptr;
        for(int i = 0; i < img_num; ++i)
        {
            Mat position_map(INPUT_WIDTH, INPUT_HEIGHT, CV_32FC3);
            outdata = &networkOut[0] + i * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
            vector<float> mydata;
            for (int j=0; j<INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS; ++j)
            {
                mydata.push_back(outdata[j] * INPUT_WIDTH * 1.1);
            }
            
            int n=0;
            for(int row=0; row<INPUT_WIDTH; row++)
            {
                for(int col=0; col<INPUT_HEIGHT; col++)
                {
                    position_map.at<Vec3f>(row,col)[2] = mydata[n];
                    ++n;
                    position_map.at<Vec3f>(row,col)[1] = mydata[n];
                    ++n;
                    position_map.at<Vec3f>(row,col)[0] = mydata[n];
                    ++n;
                }
            }
            tmpname = img_name[i];
            if (access(result_path.c_str(),6) == -1)
            {
                mkdir(result_path.c_str(), S_IRWXU);
            }
            cv::imwrite(result_path + "/"+ tmpname, position_map);
        }
        
        return landmark_status_success;
    }
    
    LandmarkStatus Resfcn::destroy()
    {
        try{
            CHECK(cudaFree(buffers[outputIndex]));
            CHECK(cudaFree(buffers[inputIndex]));
        }catch(...)
        {
            return landmark_status_cuda_free_error;
        }
        context->destroy();
        parser->destroy();
        engine->destroy();
        delete this;
        return landmark_status_success;
    }
    
    ICudaEngine* Resfcn::loadModelAndCreateEngine(const char* uffFile, int maxBatchSize, IHostMemory*& trtModelStream)
    {
        IBuilder* builder = createInferBuilder(gLogger);
        INetworkDefinition* network = builder->createNetwork();
        
        std::cout << "Begin parsing model..." << std::endl;
        if(!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
			exit(1);
            //RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
        std::cout << "End parsing model..." << std::endl;
        
        /* we create the engine */
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(MAX_WORKSPACE);
        
        std::cout << "Begin building engine..." << std::endl;
        ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (!engine)
            exit(1);
        std::cout << "End building engine..." << std::endl;
        
        /* we can clean the network and the parser */
        network->destroy();
        builder->destroy();
        trtModelStream = engine->serialize();
        return engine;
    }
    
    LandmarkStatus Resfcn::doInference(float* inputData, float* outputData, int batchSize)
    {
        int nbBindings = engine->getNbBindings();
        /*point to the input and output node*/
        size_t memSize =INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float);
        //std::vector<void*> buffers(nbBindings);
        std::vector<std::pair<int64_t, nvinfer1::DataType>> buffersSizes = calculateBindingBufferSizes(nbBindings, batchSize);
        int bindingIdxInput = 0;
        for (int i = 0; i < nbBindings; ++i)
        {
            if (engine->bindingIsInput(i))
                bindingIdxInput = i;
            else
            {
                auto bufferSizesOutput = buffersSizes[i];
                try 
		{
                    buffers[i] = safeCudaMalloc(bufferSizesOutput.first * samples_common::getElementSize(bufferSizesOutput.second));
                }catch(...)
                {
                    return landmark_status_cuda_malloc_error;
                }
            }
        }
        
        auto bufferSizesInput = buffersSizes[bindingIdxInput];
        inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
        outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	    
	try
	{
        float *tmpdata = (float *)malloc(BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS) * sizeof(float));;
	}catch(...)
	{
	    return landmark_status_host_malloc_error;
	}
        for (int i = 0; i < iteration; i++)
        {
            float total = 0, ms;
            for (int run = 0; run < run_num; run++)
            {
                /*create space for input and set the input data*/
                tmpdata = &inputData[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
                try
                {
                    buffers[bindingIdxInput] = safeCudaMalloc(bufferSizesInput.first * samples_common::getElementSize(bufferSizesInput.second));
                }catch(...)
                {
                    return landmark_status_cuda_malloc_error;
                }
                CHECK(cudaMemcpyAsync(buffers[inputIndex],tmpdata, batchSize * INPUT_CHANNELS * INPUT_WIDTH * INPUT_HEIGHT * sizeof(float), cudaMemcpyHostToDevice));
               
		auto t_start = std::chrono::high_resolution_clock::now();
                context->execute(batchSize, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
                total += ms;
		    
                for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
                {
                    if (engine->bindingIsInput(bindingIdx))
                        continue;
                    auto bufferSizesOutput = buffersSizes[bindingIdx];
                }
                total /= run_num;
                std::cout << "Average over " << run_num << " runs is " << total << " ms." << std::endl;
                tmpdata = &outputData[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
                CHECK(cudaMemcpyAsync(tmpdata, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
            }
        }
    }
    
    void* Resfcn::safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
            std::cerr << "Out of memory..." << std::endl;
			exit(1);
        }
        return deviceMem;
    }
    
    vector<std::pair<int64_t, nvinfer1::DataType>>
    Resfcn::calculateBindingBufferSizes(int nbBindings, int batchSize)
    {
        std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
        for (int i = 0; i < nbBindings; ++i)
        {
            Dims dims = engine->getBindingDimensions(i);
            nvinfer1::DataType dtype = engine->getBindingDataType(i);
            
            int64_t eltCount = samples_common::volume(dims) * batchSize;
            sizes.push_back(std::make_pair(eltCount, dtype));
        }
        
        return sizes;
    }
}

