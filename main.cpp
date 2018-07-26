//
//  main.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/3.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Resfcn.hpp"
#include "Net.hpp"
#include "util.hpp"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cudnn.h>
#include <vector>
#include <dirent.h>
#include <io.h>
#include <unistd.h>
#include <dirent.h>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "NvUtils.h"
#include "common.h"
#include "NvInfer.h"
#include "NvUffParser.h"
using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
using namespace Landmark;
using namespace cv;

const string ImagePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/image";
const string boxPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/box_api.txt";
const string faceIndex = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/face_ind.txt";
const string uv_kpt_ind = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/uv_kpt_ind.txt";
const string plotPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/plot_kpt";
const string pose_save = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/pose.txt";
const string canonical_vertices = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/canonical_vertices.txt";
const string face_detection = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/face_detection.txt";
const string json_result_path = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/landmark.json";
const int gpuID = 0;
const int batchSize = 1;
const int MaxBatchSize = 10;
const int iteration = 1;
const int inputShape[3] = {3, 256, 256};
const string suffix = ".*.jpg";
const char* OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
const char* INPUT_BLOB_NAME = "Placeholder";
const char* UFF_MODEL_PATH = "/workspace/run/xyx/TensorRT-4.0.1.6/data/landmark/face.pb.uff";


int main(int argc, const char * argv[]) {
    const int resolution = inputShape[1];
    LandmarkStatus status;
   
    Net *resfcn = createResfcn(batchSize, inputShape);
    //init inference
    status = resfcn->init(gpuID, batchSize, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, UFF_MODEL_PATH, MaxBatchSize);
    if(status != landmark_status_success)
    {
        cerr << "Init Resfcn failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;

    }
    
    vector<string> files;
    vector<string> split_result;
    vector<IMAGE> imgs;
    vector<LANDMARK> landmark;
    IMAGE tmp_img;
    files = get_all_files(ImagePath, suffix);
    if(files.size() == 0)
    {
        cerr<<"-----no image data-----"<<endl;
        exit(1);
    }
    
    int rounds = files.size() / batchSize;
    for(int i = 0; i < rounds; i++)
    {
        for(int j = 0; j < batchSize; j++)
        {
            cv::Mat img = cv::imread(files[i * batchSize + j]);
            split_result = my_split(files[i],"/");
            tmp_img.name = split_result[split_result.size()-1];
            tmp_img.img = img;
            if (!img.data)
            {
                cerr << "Read image " << files[i * batchSize + j] <<" error, No Data!" << endl;
                continue;
            }
            imgs.push_back(tmp_img);
        }
        //一个batch做一次predict
        status = resfcn->predict(imgs, face_detection, uv_kpt_ind, faceIndex, canonical_vertices, resolution, suffix, iteration, landmark, json_result_path);
        if(status != landmark_status_success)
        {
            cerr << "Resfcn predict error"
            << "\t"
            << "exit code: " << status << endl;
            return -1;
        }
    }
    
    status = resfcn->destroy();
    if(status != landmark_status_success)
    {
        cerr << "Resfcn destory error"
             << "\t"
             << "exit code: " << status << endl;
        return -1;

    }
    //保存landmark画图结果，可用于验证
    for(uint i = 0; i < files.size(); i++)
    {
        
        plot_landmark(imgs[i].img, imgs[i].name, landmark[i].landmark, plotPath);
    }
    
    return 0;
}
