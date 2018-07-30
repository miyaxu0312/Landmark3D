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
    enum ShadowStatus
    {
        shadow_status_success = 200,
        shadow_status_set_gpu_error = 501,
        shadow_status_host_malloc_error = 502,
        shadow_status_cuda_malloc_error = 503,
        shadow_status_create_tream_error = 504,
        shadow_status_deserialize_error = 505,
        shadow_status_blobname_error = 506,
        shadow_status_batchsize_exceed_error = 507,
        shadow_status_batchsize_zero_error = 508,
        shadow_status_cuda_memcpy_error = 509,
        shadow_status_invalid_gie_file = 510,
        shadow_status_cuda_free_error = 511,
        shadow_status_data_error = 512,
        shadow_status_invalid_uff_file = 513,
        shadow_status_create_model_error = 514,
    };
    enum InterMethod
    {
        nearest = 0,
        bilinear = 1,
    };

    class Net{
    public:
        virtual ShadowStatus init(const int gpuID, void *data, const int batchSize) = 0;
        virtual ShadowStatus predict(const std::vector<cv::Mat> &imgs, const std::vector<std::string> &attributes, std::vector<std::string> &results) = 0;
        virtual ShadowStatus destroy() = 0;
    };
    Shadow::Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method = bilinear);
}
#endif /* Net_hpp */
