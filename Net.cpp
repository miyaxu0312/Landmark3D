//
//  Net.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Net.hpp"
#include "Resfcn.hpp"

namespace Shadow {
    Net *createNet(int batchSize, const int *inputShape, float *preParam, InterMethod method)
    {
        Resfcn *net = new Resfcn(batchSize, inputShape, preParam, method);
        return net;
    }
}
