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
#include <regex>
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

const string ImagePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/image";
const string netOutPath="/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/network_output";
const string postPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/post";
const string boxPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/box_api.txt";
const string faceIndex = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/face_ind.txt";
const string uv_kpt_ind = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/uv_kpt_ind.txt";
const string savePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/crop_image";
const string plotPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/plot_kpt";
const string pose_save = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/pose.txt";
const string canonical_vertices = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/3DLandmark/canonical_vertices.txt";

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
    vector<Affine_Matrix> affine_matrix;
    
    pre_process(ImagePath, boxPath, netOutPath, postPath, uv_kpt_ind, faceIndex, savePath, resolution, affine_matrix, suffix);
    
    Net *resfcn = createResfcn(batchSize, inputShape);
    
    status = resfcn->init(gpuID, batchSize, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, UFF_MODEL_PATH, MaxBatchSize);
    if(status != landmark_status_success)
    {
        cerr << "Init Resfcn failed"
             << "\t"
             << "exit code: " << status << endl;
        return -1;

    }
    status = resfcn->predict(ImagePath, netOutPath, suffix, iteration, affine_matrix);
    if(status != landmark_status_success)
    {
        cerr << "Resfcn predict error"
             << "\t"
             << "exit code: " << status << endl;
        return -1;

    }
    status = resfcn->destroy();
    if(status != landmark_status_success)
    {
        cerr << "Resfcn destory error"
             << "\t"
             << "exit code: " << status << endl;
        return -1;

    }
   
    post_process(ImagePath, netOutPath, postPath, pose_save, canonical_vertices, faceIndex, uv_kpt_ind, resolution, affine_matrix, plotPath,suffix);
    
    return 0;
}
