//
//  util.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "util.hpp"
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

namespace Landmark {
void pre_process(string filePath, string boxPath, string netOutPath, string postPath, string uv_kpt_ind,string faceIndex, string savePath, int resolution, vector<Affine_Matrix> &affine_matrix, string suffix)
{
    vector<string> files;
    vector<string> split_result;
    vector<int> box;
    Mat img;
    files = get_all_files(filePath, suffix);
    cout<<"-----image num:-----"<<files.size()<<endl;
    Affine_Matrix tmp_affine_mat;
    if(files.size() == 0)
    {
        cerr<<"-----no image data-----"<<endl;
        exit(1);
    }
    for(uint i=0;i<files.size();++i)
    {
        split_result = my_split(files[i],"/");
        string name = split_result[split_result.size()-1];
        img = imread(files[i], CV_LOAD_IMAGE_UNCHANGED); // 读取每一张图片
        
        Mat similar_img;
        box = get_box(boxPath, name, suffix);
        int old_size = (box[1] - box[0] + box[3] - box[2])/2;
        int size = old_size * 1.58;
        float center_x = 0.0, center_y = 0.0;
        box[3] = box[3]- old_size * 0.3;
        box[1] = box[1] - old_size * 0.25;
        box[0] = box[0] + old_size * 0.2;
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
        /*save pre-processed image for the network*/
        if (access(savePath.c_str(),6)==-1)
        {
            mkdir(savePath.c_str(), S_IRWXU);
        }
        imwrite(savePath+"/" + name,similar_img);
        tmp_affine_mat.name = name;
        tmp_affine_mat.affine_mat = affine_mat;
        tmp_affine_mat.crop_img = similar_img;
        affine_matrix.push_back(tmp_affine_mat);
    }
}

void post_process(string ori_path, string filePath, string save_path, string pose_save, string canonical_vertices_path, string faceIndex, string uv_kpt_ind_path, int resolution, vector<Affine_Matrix> &affine_matrix,string plot_path, string suffix)
{
    vector<string> files;
    vector<string> split_result;
    vector<float> face_ind;
    Mat img, ori_img, z, vertices_T, stacked_vertices, affine_mat_stack;
    Mat pos(resolution,resolution,CV_8UC3);
    string name;
    files = get_all_files(filePath, suffix);
    for(uint i=0; i<files.size(); ++i)
    {
        string tmp = "";
        bool isfind = false;
        img = imread(files[i], IMREAD_UNCHANGED); // 读取每一张图片
        split_result = my_split(files[i], "/");
        name = split_result[split_result.size()-1]; //获取图片名
        Mat affine_mat,affine_mat_inv;
        vector<Affine_Matrix>::iterator mat_iter;
        for(mat_iter = affine_matrix.begin(); mat_iter!= affine_matrix.end(); ++mat_iter)
        {
            if((*mat_iter).name == name)
            {
                affine_mat =  (*mat_iter).affine_mat;
                invertAffineTransform(affine_mat, affine_mat_inv);
                isfind = true;
            }
        }
        if( !isfind )
            continue;
        ori_img = imread(ori_path + "/" + name, IMREAD_UNCHANGED);    //加载原始图片，方便画landmark
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
        
        for (uint row = 0; row < pos.rows; ++row)
        {
            for (uint col = 0; col < pos.cols; ++col)
            {
                pos2.at<Vec3d>(row,col)[0] = pos.at<Vec3d>(row,col)[2];
                pos2.at<Vec3d>(row,col)[1] = pos.at<Vec3d>(row,col)[1];
                pos2.at<Vec3d>(row,col)[2] = pos.at<Vec3d>(row,col)[0];
            }
            
        }
        if (access(save_path.c_str(),6)==-1)
        {
            mkdir(save_path.c_str(), S_IRWXU);
        }
        imwrite(save_path + "/" + name,pos2);
        
        ifstream f;
        f.open(faceIndex);
        if(!f)
        {
            cerr<<"-----face index file do not exist!-----"<<endl;
            exit(1);
        }
        
        while(getline(f, tmp))
        {
            istringstream iss(tmp);
            float num;
            iss >> num;
            face_ind.push_back(num);
        }
        //face index data load
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
        //kpt index data
        f.close();
        vector<vector<float>> all_vertices = get_vertices(pos2, face_ind, resolution);
        vector<vector<float>> landmark = get_landmark(pos2, uv_kpt_ind1, uv_kpt_ind2);
        //get landmark
        plot_landmark(ori_img, name, landmark, plot_path);
        vector<float> pose = estimate_pose(all_vertices, canonical_vertices_path);
        //estimate pose
        ofstream outfile(pose_save, ios::app);
        outfile<<"name:"<<name<<"\n";
        vector<float>::iterator iter;
        outfile<<"pose: ";
        for(iter=pose.begin();iter!=pose.end();iter++)
        {
            outfile<<*iter<<",";
        }
        outfile<<"\n";
        outfile.close();
    }
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

vector<int> get_box(string path, string name, string suffix)
{
    vector<string> box;
    vector<int> box_int;
    ifstream f;
    f.open(path);
    string line;               //保存读入的每一行
    while(getline(f,line))
    {
        if (line + suffix == name)
        {
            getline(f,line);
            box = my_split(line,",");
            vector<string>::iterator iter;
            for(iter=box.begin();iter!=box.end();++iter)
            {
                box_int.push_back(stoi(*iter));
            }
        }
        else{
            getline(f,line);
        }
    }
    return box_int;
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
    
vector<vector<float>> get_landmark(Mat pos, vector<float> uv_kpt_ind_0,vector<float> uv_kpt_ind_1)
{
    vector<vector<float>> landmark(68,vector<float>(3,0));
    for (uint i=0; i<uv_kpt_ind_0.size();++i)
    {
        landmark[i][0] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[2];
        landmark[i][1] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[1];
        landmark[i][2] = pos.at<Vec3d>(uv_kpt_ind_1[i],uv_kpt_ind_0[i])[0];
    }

    return landmark;
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
