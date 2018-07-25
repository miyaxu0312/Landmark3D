//
//  Net.cpp
//  landmark
//
//  Created by xuyixuan on 2018/7/24.
//  Copyright © 2018年 xuyixuan. All rights reserved.
//

#include "Net.hpp"
#include "Resfcn.hpp"

namespace Landmark {
    Net *createResfcn(const int batchSize, const int *inputShape)
    {
        Resfcn *resfcn = new Resfcn(batchSize, inputShape);
        return resfcn;
    }
}
