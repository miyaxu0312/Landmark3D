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

	struct IMAGE
	{
	     string name;
		 Mat img;
	};
	struct LANDMARK
	{

		string name;
		vector<vector<float>> landmark;
	};
	struct Affine_Matrix
	{
		string name;
		Mat affine_mat;
		Mat crop_img;
	};

    vector<string> get_all_files(string path, string suffix);
    vector<string> my_split(string my_str, string seperate);
    bool searchkey(vector<int> a, int value);
    void getFromText(String nameStr, Mat &myMat);
    void get_box(string path, string img_name,int resolution, bool &isfind, vector<int> &box);
    void plot_landmark(Mat &image, string name, vector<vector<float>> &kpt, string plot_path);
    void get_vertices(Mat &pos, vector<float> face_ind, int resolution,vector<vector<float>> &all_veritices);
    vector<LANDMARK> get_landmark(Mat &pos, string name, vector<float> uv_kpt_ind_0,vector<float> uv_kpt_ind_1, vector<vector<float>> &landmark_one);
    void estimate_pose(vector<vector<float>> &vertices, string canonical_vertices_path, vector<float> &pose);
    Mat P2sRt(Mat p);
    void matrix2angle(Mat &p, vector<float> &pose);
    void parse_request_boxes(string &attribute, int resolution, bool &isfind, vector<int> &box);
}

#endif /* util_hpp */
