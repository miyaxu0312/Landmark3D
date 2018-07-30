//
//  Net.hpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#ifndef Net_hpp
#define Net_hpp
#include "util.hpp"
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

namespace  Shadow {
    enum LandmarkStatus
    {
        landmark_status_success = 200,
        landmark_status_set_gpu_error = 501,
        landmark_status_host_malloc_error = 502,
        landmark_status_cuda_malloc_error = 503,
        landmark_status_create_model_error = 504,
        landmark_status_deserialize_error = 505,
        landmark_status_blobname_error = 506,
        landmark_status_batchsize_exceed_error = 507,
        landmark_status_cuda_memcpy_error = 508,
        landmark_status_invalid_uff_file = 509,
        landmark_status_cuda_free_error = 510,
        landmark_status_data_error = 511,
    };
    enum InterMethod
    {
        nearest = 0,
        bilinear = 1,
    };

    class Net{
    public:
        virtual LandmarkStatus init(const int gpuID, void *data, const int batchSize) = 0;
        virtual LandmarkStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results) = 0;
        virtual LandmarkStatus destroy() = 0;
    };
    Shadow::Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method = bilinear);
}
#endif /* Net_hpp */
