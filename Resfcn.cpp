//
//  Resfcn.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Resfcn.hpp"
#include "util.hpp"
#include "rapidJson/document.h"
#include "rapidJson/stringbuffer.h"
#include "rapidJson/writer.h"
#include <vector>
#include <dirent.h>
#include <io.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
using namespace cv;
using namespace rapidjson;
namespace Landmark {
static Logger gLogger;
static samples_common::Args args;
#define MAX_WORKSPACE (1<<30)

    Resfcn::Resfcn(const int batchSize, const int *inputShape)
    {
        this->BATCH_SIZE = batchSize;
        this->INPUT_CHANNELS = inputShape[0];
        this->INPUT_WIDTH = inputShape[1];
        this->INPUT_HEIGHT = inputShape[2];
    }
    
    LandmarkStatus Resfcn::init(const int gpuID, const int batchSize, const char* INPUT_BLOB_NAME, const char* OUTPUT_BLOB_NAME, const char* UFF_MODEL_PATH, const int MaxBatchSize,int n)
    {
        this->INPUT_BLOB_NAME = INPUT_BLOB_NAME;
        this->OUTPUT_BLOB_NAME = OUTPUT_BLOB_NAME;
        img_num = batchSize;
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
    
    vector<float> Resfcn::pre_process(const string boxPath, int resolution, const string suffix)
    {
        vector<string> files;
        vector<string> split_result;
        vector<int> box;
        Mat img;
        string name;
        Affine_Matrix tmp_affine_mat;
        vector<float> data;
	    
        for(int i = 0;i < img_num; ++i)
        {
            name = ori_img[i].name;
            img = ori_img[i].img;   // 原图
            
            Mat similar_img;
            bool isfind = false;
	
            get_box(boxPath, name, INPUT_WIDTH, isfind, box);
			cout<<box[0]<<","<<box[1]<<","<<box[2]<<","<<box[3]<<endl;
            int old_size = (box[1] - box[0] + box[3] - box[2])/2;
            int size = old_size * 1.58;
            float center_x = 0.0, center_y = 0.0;
            box[3] = box[3]- old_size * 0.3;
            box[1] = box[1] - old_size * 0.25;
            box[0] = box[0] + old_size * 0.25;
            center_x = box[1] - (box[1] - box[0]) / 2.0;
            center_y = box[3] - (box[3] - box[2]) / 2.0 + old_size * 0.14;
            
            float temp_src[3][2] = {{center_x-size/2, center_y-size/2},{center_x - size/2, center_y + size/2},{center_x+size/2, center_y-size/2}};
            
            Mat srcMat(3, 2, CV_32F,temp_src);
            float temp_dest[3][2] = {{0, 0},{0, static_cast<float>(resolution-1)},{static_cast<float>(resolution-1), 0}};
            Mat destMat(3, 2, CV_32F,temp_dest);
            Mat affine_mat = getAffineTransform(srcMat, destMat);
			
            img.convertTo(img,CV_32FC3);
            img = img/255.;
            
            warpAffine(img, similar_img, affine_mat,  similar_img.size());
           
            //will be used in post-processed stage
            tmp_affine_mat.name = name;
            tmp_affine_mat.affine_mat = affine_mat;
            tmp_affine_mat.crop_img = similar_img;
            this->affine_matrix.push_back(tmp_affine_mat);//
            int num = 0;
            img.convertTo(img, CV_32FC3);
            for(int c=0; c<INPUT_CHANNELS; ++c)
            {
                for(int row=0; row<INPUT_WIDTH; row++)
                {
                    for(int col=0; col<INPUT_HEIGHT; col++, ++num)
                    {
                        data.push_back(similar_img.at<Vec3f>(row,col)[c]);
                    }
                }
            }
        }
        return data;
    }
    
    LandmarkStatus Resfcn::predict(vector<IMAGE> &imgs, const string boxPath, const string uv_kpt_ind, const string faceIndex, const string canonical_vertices_path, int resolution, const string suffix, const int iteration, string json_result_path)
    {
        //vector<IMAGE> imgs = *(imgs2);
		vector<IMAGE>().swap(ori_img);
        vector<IMAGE>().swap(network_out);
        vector<IMAGE>().swap(position_map);
        vector<Affine_Matrix>().swap(affine_matrix);
		Mat img;            //一个batch一次
        this->run_num = img_num;
        this->iteration = iteration;
        this->ori_img.assign(imgs.begin(),imgs.end());
        vector<float> networkOut(BATCH_SIZE * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH);
         //存储一个batch的仿射矩阵

        string tmpname;
        cout<<"prepare pre_process..."<<endl;
	    cout<<"num:----"<<imgs.size()<<"-----";
        vector<float> data = pre_process(boxPath, resolution, suffix);
        cout<<"complete pre_process..."<<endl;
        LandmarkStatus status = doInference(&data[0], &networkOut[0], BATCH_SIZE);
        
        if (status != landmark_status_success)
        {
            cerr << "Resfcn predict failed"
             << "\t"
             << "exit code: " << status << endl;
            return status;
        }

        std::cout<<"Inference uploaded..."<<endl;
        
        float* outdata=nullptr;
        IMAGE tmp_img;
        for(int i = 0; i < img_num; ++i)
        {
			cout<<"save..."<<endl;
            Mat network_out_img(INPUT_WIDTH, INPUT_HEIGHT, CV_32FC3);
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
                    network_out_img.at<Vec3f>(row,col)[2] = mydata[n];
                    ++n;
                    network_out_img.at<Vec3f>(row,col)[1] = mydata[n];
                    ++n;
                    network_out_img.at<Vec3f>(row,col)[0] = mydata[n];
                    ++n;
                }
            }
            tmp_img.img = network_out_img;
            tmp_img.name = ori_img[i].name;
            network_out.push_back(tmp_img);
        }
		cout<<"begin post_processed..."<<endl;
        post_process(canonical_vertices_path, faceIndex, uv_kpt_ind, resolution,suffix, json_result_path);
        return landmark_status_success;
    }
    
    void Resfcn::post_process(const string canonical_vertices_path, const string faceIndex, const string uv_kpt_ind_path, int resolution, const string suffix, string json_result_path)
    {
        vector<float> face_ind;
        
        Mat img, z, vertices_T, stacked_vertices, affine_mat_stack;
        Mat pos(resolution, resolution, CV_8UC3);
        string name;
        for(uint i=0; i < img_num; ++i)
        {
            string tmp = "";
            img = network_out[i].img;
            name = network_out[i].name;
            Mat affine_mat,affine_mat_inv;
            
            //img = affine_matrix[i].crop_img;
            affine_mat = affine_matrix[i].affine_mat;
            invertAffineTransform(affine_mat, affine_mat_inv);
            
            Mat cropped_vertices(resolution*resolution,3,img.type()), cropped_vertices_T(3,resolution*resolution,img.type());
            
            cropped_vertices = img.reshape(1, resolution * resolution);
            Mat cropped_vertices_swap(resolution*resolution,3,cropped_vertices.type());
            
            cropped_vertices.col(0).copyTo(cropped_vertices_swap.col(2));
            cropped_vertices.col(1).copyTo(cropped_vertices_swap.col(1));
            cropped_vertices.col(2).copyTo(cropped_vertices_swap.col(0));
            
            transpose(cropped_vertices_swap, cropped_vertices_T);
            cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());
            z = cropped_vertices_T.row(2).clone() / affine_mat.at<double>(0,0);
            
            Mat ones_mat(1,resolution*resolution,cropped_vertices_T.type(),Scalar(1));
            ones_mat.copyTo(cropped_vertices_T.row(2));
            
            cropped_vertices_T.convertTo(cropped_vertices_T, affine_mat.type());
            
            Mat vertices =  affine_mat_inv * cropped_vertices_T;
            z.convertTo(z, vertices.type());
            
            vconcat(vertices.rowRange(0, 2), z, stacked_vertices);
            transpose(stacked_vertices, vertices_T);
            pos = vertices_T.reshape(3,resolution);
            Mat pos2(resolution,resolution,CV_64FC3);
            
            for (int row = 0; row < pos.rows; ++row)
            {
                for (int col = 0; col < pos.cols; ++col)
                {
                    pos2.at<Vec3d>(row,col)[0] = pos.at<Vec3d>(row,col)[2];
                    pos2.at<Vec3d>(row,col)[1] = pos.at<Vec3d>(row,col)[1];
                    pos2.at<Vec3d>(row,col)[2] = pos.at<Vec3d>(row,col)[0];
                }
                
            }
            IMAGE tmp_position_map;
            tmp_position_map.name = name;
            tmp_position_map.img = img;
            position_map.push_back(tmp_position_map);
			cout<<"post_processed completed..."<<endl;
            //一个batch处理一次结果
            dealResult(resolution, faceIndex, uv_kpt_ind_path, json_result_path,canonical_vertices_path);
		}
    }
        
    //deal with position map
    void Resfcn::dealResult(const int resolution, const string faceIndex, const string uv_kpt_ind_path, string json_result_path, string canonical_vertices_path)
        {
            ifstream f;
            f.open(faceIndex);
            if(!f)
            {
                cerr<<"-----face index file do not exist!-----"<<endl;
                exit(1);
            }
            string tmp;
			vector<float> face_ind;
            while(getline(f, tmp))
            {
                istringstream iss(tmp);
                float num;
                iss >> num;
                face_ind.push_back(num);
            }
            cout<<"face index data load"<<endl;
            f.close();
            
            f.open(uv_kpt_ind_path);
            if (!f)
            {
                cerr<<"-----uv kpt index file do not exist!-----"<<endl;
                exit(1);
            }
            getline(f, tmp);
            vector<string> all_uv = my_split(tmp, " ");
            vector<string>::iterator uv_iter;
            int ind_num=1;
            vector<float> uv_kpt_ind1,uv_kpt_ind2;
            
            for (uv_iter=all_uv.begin(); uv_iter!=all_uv.end(); ++uv_iter, ++ind_num)
            {
                istringstream iss(*uv_iter);
                float num;
                iss >> num;
                if (ind_num <= 68 && ind_num > 0)
                    uv_kpt_ind1.push_back(num);
                else if(ind_num > 68 && ind_num <= 68*2)
                    uv_kpt_ind2.push_back(num);
            }
            cout<<"kpt index data"<<endl;
            f.close();
            for(uint i = 0;i < img_num; i++)
            {
			    vector<vector<float>> all_vertices(face_ind.size(),vector<float>(3,0));
				get_vertices(position_map[i].img, face_ind, resolution, all_vertices); //一个batch的点
				cout<<"get_veritices.."<<endl;
                vector<vector<float>> landmark_one(68, vector<float>(3,0));
				this->landmark = get_landmark(position_map[i].img, position_map[i].name, uv_kpt_ind1, uv_kpt_ind2,landmark_one);
                cout<<"get landmark"<<endl;
                vector<float> pose(3,0);
				estimate_pose(all_vertices, canonical_vertices_path, pose);
				cout<<"begin get result json..."<<endl;
                getResultJson(landmark_one, pose, position_map[i].name, json_result_path);
            }
        }
    
    void Resfcn::getResultJson(vector<vector<float>> &landmark_one, vector<float> &pose, string name, string json_result_path)
    {
        Document document;
        auto &alloc = document.GetAllocator();
        Value json_result(kObjectType), j_landmark(kArrayType), j_pos(kArrayType);
        for(uint i = 0; i < 68; i++)
        {
            Value points(kArrayType);
            points.PushBack(Value(landmark_one[i][0]), alloc).PushBack(Value(landmark_one[i][1]), alloc).PushBack(Value(landmark_one[i][2]), alloc);
            j_landmark.PushBack(points,alloc);
        }
        json_result.AddMember("landmark", j_landmark, alloc);
        j_pos.PushBack(Value(pose[0]), alloc).PushBack(Value(pose[1]), alloc).PushBack(Value(pose[2]), alloc);
        json_result.AddMember("pose", j_pos, alloc);
        
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        json_result.Accept(writer);
        string results = string(buffer.GetString());
        
        ofstream outfile(json_result_path, ios::app);
        outfile<<results;
        outfile<<"\n";
        outfile.close();
		cout<<"end..."<<endl;
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
        
        //std::cout << "Begin parsing model..." << std::endl;
        if(!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
	{
	    cerr<<"fail to parse uff file"<<endl;
	    exit(1);
	}
        //std::cout << "End parsing model..." << std::endl;
        
        /* we create the engine */
        builder->setMaxBatchSize(maxBatchSize);
        builder->setMaxWorkspaceSize(MAX_WORKSPACE);
        
        //std::cout << "Begin building engine..." << std::endl;
        ICudaEngine* engine = builder->buildCudaEngine(*network);
        if (!engine)
            exit(1);
        //std::cout << "End building engine..." << std::endl;
        
        /* we can clean the network and the parser */
        network->destroy();
        builder->destroy();
        trtModelStream = engine->serialize();
        return engine;
    }
    
    LandmarkStatus Resfcn::doInference(float* inputData, float* outputData, int batchSize)
    {
        int nbBindings = engine->getNbBindings();
        size_t memSize =INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS * sizeof(float);
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
        float *tmpdata;
        try
        {
            tmpdata = (float *)malloc(BATCH_SIZE * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS* sizeof(float));
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
                total /= run_num;
                cout << "Average over " << run_num << " runs is " << total << " ms." << std::endl;
                tmpdata = &outputData[0] + run * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
                CHECK(cudaMemcpyAsync(tmpdata, buffers[outputIndex], memSize, cudaMemcpyDeviceToHost));
            }
        }
	return landmark_status_success;
    }
    
    void* Resfcn::safeCudaMalloc(size_t memSize)
    {
        void* deviceMem;
        CHECK(cudaMalloc(&deviceMem, memSize));
        if (deviceMem == nullptr)
        {
            cerr << "Out of memory..." << std::endl;
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

