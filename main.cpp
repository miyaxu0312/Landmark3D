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
using namespace Shadow;
using namespace cv;

const string ImagePath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/Landmark3D/image";
const string plotPath = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/Landmark3D/plot_kpt";
const string pose_save = "/workspace/run/xyx/TensorRT-4.0.1.6/samples/Landmark3D/pose.txt";

float preParam[4] = {103.52, 116.28, 123.675, 0.017};
const int gpuID = 0;
const int batchSize = 1;
const int MaxBatchSize = 10;
const int iteration = 1;
const int inputShape[3] = {3, 256, 256};
const string suffix = ".*.jpg";
static const char* OUTPUT_BLOB_NAME = "resfcn256/Conv2d_transpose_16/Sigmoid";
static const char* INPUT_BLOB_NAME = "Placeholder";
static const char* UFF_MODEL_PATH = "/workspace/run/xyx/TensorRT-4.0.1.6/data/landmark/face.pb.uff";

vector<string> face_detection_string = {
	\"detections": [{"index": 1, "score": 0.9999485015869141, "pts": [[133, 114], [341, 114], [341, 364], [133, 364]], "class": "face"}],
	
	 \"detections": [{"index": 1, "score": 0.9969366788864136, "pts": [[31, 18], [178, 18], [178, 203], [31, 203]], "class": "face"}],
	
	 \"detections": [{"index": 1, "score": 0.9934788942337036, "pts": [[44, 54], [150, 54], [150, 183], [44, 183]], "class": "face"}],
	
	\ "detections": [{"index": 1, "score": 0.99978107213974, "pts": [[97, 131], [246, 131], [246, 319], [97, 319]], "class": "face"}],
	
	  \"detections": [{"index": 1, "score": 0.9999821186065674, "pts": [[116, 111], [302, 111], [302, 355], [116, 355]], "class": "face"}],
	
	 \ "detections": [{"index": 1, "score": 0.999951958656311, "pts": [[38, 49], [181, 49], [181, 234], [38, 234]], "class": "face"}]
};

int main(int argc, const char * argv[]) {
    
    ShadowStatus status;
    vector<string> files;
    vector<string> results;
    files = get_all_files(ImagePath,suffix);
    Net *resfcn = createNet(batchSize, inputShape, preParam);
    //init inference
    status = resfcn->init(gpuID, nullptr, batchSize);
    if(status != shadow_status_success)
    {
        cerr << "Init Resfcn failed"
        << "\t"
        << "exit code: " << status << endl;
        return -1;
        
    }
    
    vector<string> split_result;
    string tmp_name;
    if(files.size() == 0)
    {
        cerr<<"-----no image data-----"<<endl;
        exit(1);
    }
    // cout<<"-----img-num-----"<<files.size();
    int rounds = files.size() / batchSize;
    for(int i = 0; i < rounds; i++)
    {
        vector<Mat> imgs;
        vector<string> imgName;
        vector<string> attributes;
        
        for(int j = 0; j < batchSize; j++)
        {
            Mat img = cv::imread(files[i * batchSize + j]);
            split_result = my_split(files[i * batchSize + j],"/");
            tmp_name = split_result[split_result.size()-1];
            imgName.push_back(tmp_name);
            attributes.push_back(face_detection_string[i]);
            if (!img.data)
            {
                cerr << "Read image " << files[i * batchSize + j] <<" error, No Data!" << endl;
                continue;
            }
            imgs.push_back(img);
            cout<<"get data"<<endl;
        }
        //一个batch做一次predict
        cout<<"data prepared..."<<endl;
        cout<<imgs.size()<<endl;
        status = resfcn->predict(imgs, attributes, results);
        if(status != shadow_status_success)
        {
            cerr << "Resfcn predict error"
            << "\t"
            << "exit code: " << status << endl;
            return -1;
        }
        
        cout<<imgs.size();
        vector<vector<float>> landmark = get_landmark_result();
        //保存landmark画图结果，可用于验证
        for(int j=0;j<batchSize;j++)
        {
            plot_landmark(imgs[j], imgName[j], landmark, plotPath);
        }
    }
    cout<<"predict completed..."<<endl;
    status = resfcn->destroy();
    if(status != shadow_status_success)
    {
        cerr << "Resfcn destory error"
        << "\t"
        << "exit code: " << status << endl;
        return -1;
        
    }
    
    return 0;
}
