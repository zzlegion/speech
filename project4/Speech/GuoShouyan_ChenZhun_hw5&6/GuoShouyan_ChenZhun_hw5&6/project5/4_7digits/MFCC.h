//
//  MFCC.h
//  speech_test
//
//  Created by shouyanguo on 9/9/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//

#ifndef __speech_test__MFCC__
#define __speech_test__MFCC__

#include <stdio.h>
#include <vector>
std::vector<float*> MFCC(short *data, int startPoint, int endPoint);

#endif /* defined(__speech_test__MFCC__) */
