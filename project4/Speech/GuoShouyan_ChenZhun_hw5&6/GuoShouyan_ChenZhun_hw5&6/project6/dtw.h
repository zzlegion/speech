//
//  dtw.h
//  speech_test
//
//  Created by shouyanguo on 10/6/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//

#ifndef __speech_test__dtw__
#define __speech_test__dtw__

#include <stdio.h>
#include <vector>
using namespace std;

vector<vector<float> > makeWord(vector<vector<float> > mfcc);
vector<vector<float> >  makeTemplet(vector<vector<float> > &mean);
vector<vector<float> > makeTrellis(vector<vector<float> >  templet, long wordLength);

float DP(vector<vector<float> >  &templet, vector<vector<float> > &words, vector<vector<float> > &score, vector<vector<float> > &vari, vector<int> &boundary,vector<vector<float> > tran);
#endif /* defined(__speech_test__dtw__) */
