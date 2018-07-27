//
//  util.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "util.hpp"
#include "rapidJson/document.h"
#include "rapidJson/stringbuffer.h"
#include "rapidJson/writer.h"
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <sstream>
#include <string.h>
#include <vector>
#include <fstream>
#include <regex>
#include <io.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "common.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace rapidjson;
namespace Landmark {
inline rapidjson::Document get_document(const std::string &json_text) {
    rapidjson::Document document;
    if (!json_text.empty()) {
        document.Parse(json_text.c_str());
    } else {
        document.Parse("{}");
    }
    return document;
}

inline float get_float(const rapidjson::Value &root, const std::string &name,
                       const float &def) {
    if (root.HasMember(name.c_str())) {
        const auto &value = root[name.c_str()];
        CHECK(value.IsNumber());
        return value.GetFloat();
    } else {
        return def;
    }
}

inline bool check_valid_box_pts(const float pts[4][2]){
    if(pts[0][0]==pts[3][0] &&
       pts[0][1]==pts[1][1] &&
       pts[1][0]==pts[2][0] &&
       pts[2][1]==pts[3][1] &&
       pts[2][0]>pts[0][0] &&
       pts[2][1]>pts[0][1]
       ){
        return true;
    }
    return false;
}
    
    
vector<string> get_all_files(string path, string suffix)
{
    vector<string> files;
    regex reg_obj(suffix, regex::icase);
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(path.c_str())) == NULL)
    {
        cerr << "can not open this file." << endl;
    }
    else{
        while((dirp = readdir(dp)) != NULL)
        {
            if(dirp->d_type == 8 && regex_match(dirp->d_name, reg_obj))
            {
                string file_absolute_path = path.c_str();
                file_absolute_path = file_absolute_path.append("/");
                file_absolute_path = file_absolute_path.append(dirp->d_name);
                files.push_back(file_absolute_path);
            }
        }
    }
    closedir(dp);
    return files;
}

vector<string> my_split(string my_str,string seperate)
{
    vector<string> result;
    size_t split_index = my_str.find(seperate);
    size_t start = 0;
    
    while(string::npos!=split_index)
    {
        result.push_back(my_str.substr(start,split_index-start));
        start = split_index+seperate.size();
        split_index = my_str.find(seperate,start);
    }
    result.push_back(my_str.substr(start,split_index-start));
    return result;
}

bool searchkey(vector<int> a, int value)
{
    for(uint i=0;i<a.size();i++)
    {
        if(a[i]==value)
            return true;
    }
    return false;
}


void getFromText(String nameStr, Mat &myMat)
{
    ifstream myFaceFile;
    myFaceFile.open(nameStr);
    if (!myFaceFile)
    {
        cerr<<"-----face index file do not exist!-----"<<endl;
        exit(1);
    }
    vector<string> result(3);
    string tmp;
    int i=0,line = 0;
    while(getline(myFaceFile, tmp))
    {
        result = my_split(tmp, ",");
        vector<string>::iterator iter;
        i = 0;
        for(iter=result.begin();iter!=result.end();++iter,i++)
        {
            istringstream iss(*iter);
            float num;
            iss >> num;
            myMat.at<float>(line,i)=num;
        }
        ++line;    
    }
    myFaceFile.close();
}

vector<int> get_box(string path, string img_name, int resolution, bool &isfind)
{
    vector<int> box;
    vector<string> tmp;
    ifstream f;
    f.open(path);
	if(!f)
	{
		cerr<<"open box file error..."<<endl;
		exit(1);
	}
    string line, json_result;               //保存读入的每一行
	isfind = true;
    while(getline(f,line) && isfind)
    {
		const auto &document = get_document(line);
        if(document.HasMember("url"))
        {
            string name = document["url"].GetString();
            tmp = my_split(name, "/");
			name = tmp[tmp.size()-1];
            if (img_name == name)
            {
                box = parse_request_boxes(line, resolution, isfind);
            }
            else{
                getline(f,line);
            }
        }
        else{
				isfind = false;
			}
        
    }
}
    
vector<int>  parse_request_boxes(string &attribute, int resolution, bool &isfind)
    {
    vector<int> box;
    const auto &document = get_document(attribute);
    char pts_valid_msg[] = ("'attribute' in request data must be a valid json dict string,"
                            " and has key 'pts'."
                            " pts must be in the form as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."
                            " all of x1, x2, y1, y2 can be parsed into int values."
                            " And also must have (x2>x1 && y2>y1).");
    
    if (!document.HasMember("result")){
        isfind = false;
    }
    // need  ignore in the post process
    if (document.HasMember("result")){
		const Value& result = document["result"];
		const Value& ptses = result["detections"];
        if (ptses.IsArray()){
        if(ptses.Size()==0)
            isfind = false;
            const auto &pts=ptses["pts"];
			cout<<pts.IsArray();
            float t_pts[4][2];
            try{
                bool isArray=pts.IsArray();
                if(!isArray){
                    isfind = false;
                }
                const int size=pts.Size();
                if(size!=4){
                    isfind = false;
                }
                for(int i=0; i<4; i++){
                    for(int j=0; j<2; j++){
						cout<<pts[i][j].GetString();
                        t_pts[i][j] = pts[i][j].GetInt();
                    }
                }
            }
            catch (...){
                isfind = false;
            }
            
            if(!check_valid_box_pts(t_pts))
            {
                isfind = false;
            }
            
            float xmin = t_pts[0][0];
            float ymin = t_pts[0][1];
            float xmax = t_pts[2][0];
            float ymax = t_pts[2][1];
        if(xmin>=0 && xmin<resolution && ymin>=0 && ymin<resolution && xmax>=0 && xmax<resolution && ymax>=0 && ymax<resolution)		{
            box.push_back(xmin);
            box.push_back(xmax);
            box.push_back(ymin);
            box.push_back(ymax);
            }else{
                isfind = false;
            }
        
        isfind = true;
		}
    }
    return box;
    }

       
void plot_landmark(Mat img, string name, vector<vector<float>> kpt, string plot_path)
{
    Mat image = img.clone();
    vector<int> end_list = {17-1, 22-1, 27-1, 42-1, 48-1, 31-1, 36-1, 68-1};
    for(uint i = 0; i < 68; i++)
    {
        int start_point_x, start_point_y, end_point_x, end_point_y;
        start_point_x = int(round(kpt[i][0]));
        start_point_y = int(round(kpt[i][1]));
        Point center1(start_point_x,start_point_y);
        circle(image, center1, 2, Scalar(0,0,255));
        
        if (searchkey(end_list,i))
            continue;
        
        end_point_x = int(round(kpt[i+1][0]));
        end_point_y = int(round(kpt[i+1][1]));
        Point center2(end_point_x,end_point_y);
        line(image, center1, center2, Scalar(0,255,0));
    }

    if (access(plot_path.c_str(),6)==-1)
    {
        mkdir(plot_path.c_str(), S_IRWXU);
    }
    imwrite(plot_path + "/" + name, image);
}
    
vector<vector<float>> get_vertices(Mat pos, vector<float> face_ind, int resolution)
{
    Mat all_vertices = pos.reshape(1,resolution*resolution);
    vector<vector<float>> result(face_ind.size(),vector<float>(3,0));
    vector<float>::iterator iter;
    int i=0;
    for (iter=face_ind.begin();iter!=face_ind.end();iter++)
    {
        result[i][0] = all_vertices.at<double>(int(*iter),2);
        result[i][1] = all_vertices.at<double>(int(*iter),1);
        result[i][2] = all_vertices.at<double>(int(*iter),0);
        i++;
    }

    return result;
}
    
vector<vector<float>> get_landmark(Mat pos, string name, vector<float> uv_kpt_ind_0,vector<float> uv_kpt_ind_1, vector<LANDMARK> &landmark)
{
    LANDMARK tmp_landmark;
    vector<vector<float>> landmark_one;
    for (uint i=0; i<uv_kpt_ind_0.size();++i)
    {
        landmark_one[i][0] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[2];
        landmark_one[i][1] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[1];
        landmark_one[i][2] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[0];
    }
    tmp_landmark.landmark = landmark_one;
    tmp_landmark.name = name;
    landmark.push_back(tmp_landmark);
    return landmark_one;
}
    
vector<float> estimate_pose(vector<vector<float>> vertices, string canonical_vertices_path)
{
    Mat canonical_vertices_homo;
    Mat canonical_vertices = Mat::zeros(131601/3, 3, CV_32FC1);
    getFromText(canonical_vertices_path, canonical_vertices);

    Mat ones_mat(131601/3,1,canonical_vertices.type(),Scalar(1));
    ones_mat.convertTo(ones_mat, CV_32F);
    hconcat(canonical_vertices, ones_mat, canonical_vertices_homo);

    Mat canonical_vertices_homo_T, vertices_T;
    CvMat *canonical_vertices_homo_T_pointer = cvCreateMat(43867, 4, CV_32FC1);
    CvMat *vertices_T_pointer = cvCreateMat(43867, 3, CV_32FC1);
    CvMat *P_pointer = cvCreateMat(4, 3,CV_32FC1);

    cvSetData(canonical_vertices_homo_T_pointer, canonical_vertices_homo.data, CV_AUTOSTEP);

    for (uint i = 0; i < 43867; i++)
    {
        for(uint j = 0; j < 3; j++)
        {
            cvmSet(vertices_T_pointer, i, j, vertices[i][j]);
        }
    }

    cvSolve(canonical_vertices_homo_T_pointer,vertices_T_pointer, P_pointer);

    Mat P(P_pointer->rows,P_pointer->cols,P_pointer->type,P_pointer->data.fl);
    Mat P_T(P_pointer->cols,P_pointer->rows,P_pointer->type);
    
    transpose(P, P_T);

    Mat rotation_matrix = P2sRt(P_T);
    vector<float> pose = matrix2angle(rotation_matrix);
    return pose;
}

//p 3*4
Mat P2sRt(Mat P)
{
    Mat t2d(2,1, P.type());
    Mat R1(1,3, P.type());
    Mat R2(1,3, P.type());
    Mat R(3,3, P.type());
    //t2d
    Mat P_row0 = P.rowRange(0,1).clone();
    R1 = P_row0.colRange(0, 3).clone();
    Mat P_row1 = P.row(1).clone();
    P_row1.colRange(0, 3).copyTo(R2);
    Mat r1 = R1 / norm(R1);
    Mat r2 = R2 / norm(R2);

    CvMat *r1_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvSetData(r1_pointer, r1.data, CV_AUTOSTEP);

    CvMat *r2_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvSetData(r2_pointer, r2.data, CV_AUTOSTEP);

    CvMat *r3_pointer=cvCreateMat(1, 3,CV_32FC1);
    cvCrossProduct(r1_pointer,r2_pointer,r3_pointer);

    Mat r3(r3_pointer->rows,r3_pointer->cols, r3_pointer->type, r3_pointer->data.fl);
    vconcat(r1, r2, R);
    vconcat(R, r3, R);
    return R;
}

//r 3*3
vector<float> matrix2angle(Mat R)
{
    vector<float> pose_angle(3,1);
    float x = 0, y = 0, z = 0;
    if (R.at<float>(2,0) != 1 || R.at<float>(2,0) != -1)
    {
        x = asin(R.at<float>(2,0));
        y = atan2(R.at<float>(2,1)/cos(x), R.at<float>(2,2)/cos(x));
        z = atan2(R.at<float>(1,0)/cos(x), R.at<float>(0,0)/cos(x));
    }
    else{
        z = 0;
        if (R.at<float>(2,0) == -1)
        {
            x = M_PI / 2;
            y = z + atan2(-R.at<float>(0,1), -R.at<float>(0,2));
        }
    }
    pose_angle.push_back(x);
    pose_angle.push_back(y);
    pose_angle.push_back(z);
    return pose_angle;
}
}
