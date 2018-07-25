//
//  util.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef util_hpp
#define util_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

namespace Landmark{
    struct Affine_Matrix
    {
        string name;
        Mat affine_mat;
        Mat crop_img;
    };
    void pre_process(const string ImagePath, const string boxPath, const string netOutPath, const string postPath, const string uv_kpt_ind, const string faceIndex, const string savePath, int resolution, vector<Affine_Matrix> &affine_matrix);
    void post_process(const string ori_path, const string filePath, const string save_path, const string pose_save, const string canonical_vertices, const string faceIndex, const string uv_kpt_ind, int resolution, vector<Affine_Matrix> &affine_matrix, const string plot_path);
    vector<string> get_all_files(string path, string suffix);
    vector<string> my_split(string my_str, string seperate);
    bool searchkey(vector<int> a, int value);
    void getFromText(String nameStr, Mat &myMat);
    vector<int> get_box(string path, string name);
    void plot_landmark(Mat image, string name, vector<vector<float>> kpt, string plot_path);
    vector<vector<float>> get_vertices(Mat pos, vector<float> face_ind, int resolution);
    vector<vector<float>> get_landmark(Mat pos, vector<float> uv_kpt_ind_0, vector<float> uv_kpt_ind_1);
    vector<float> estimate_pose(vector<vector<float>> vertices, string canonical_vertices_path);
    Mat P2sRt(Mat p);
    vector<float> matrix2angle(Mat p);
}

#endif /* util_hpp */
