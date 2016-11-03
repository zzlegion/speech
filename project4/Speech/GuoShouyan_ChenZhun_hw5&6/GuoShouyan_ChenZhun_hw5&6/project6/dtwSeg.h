//
//  dtwSeg.h
//
//  Created by shouyanguo on 10/6/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//
// used for digit level segmentation
// just concatenate (trained) mean/vari/tran according to the digit sequence 


#ifndef __speech_dtw__seg__
#define __speech_dtw__seg__

#include <stdio.h>
#include <vector>
using namespace std;

void SEGprintResult(int digit, vector<vector<float> > meanP,
	vector<vector<float> > variP,vector<vector<float> > tranP);

// read the trained mfcc(single Gaussian) of each digit (0~9)
vector<vector<vector<float> > >  SEGreadTemplet(vector<string> &fileName);
vector<vector<vector<float> > >  SEGreadVari(vector<string> &fileName);
vector<vector<vector<float> > >  SEGreadTran(vector<string> &fileName);

vector<vector<float> >  SEGmakeWord(vector<vector<float> > &mfcc);
vector<vector<float> >  SEGmakeTemplet(vector<vector<vector<float> > > mean, vector<int> digit );
vector<vector<float> >  SEGmakeTran(vector<vector<vector<float> > > tran, vector<int> digit );


void SEG_DP(vector<vector<float> >  &templet, vector<vector<float> > &words, vector<vector<float> > &trellis, vector<vector<float> > &vari,vector<vector<float> > &tran);

vector<vector<float> > SEGmakeTrellis(vector<vector<float> >  templet, long wordLength);

void SEGtraceBack(vector<vector<float> > &trellis,
               vector<vector<vector<vector<float> > > > &segmented,
               vector<int> &boundaryOfT, vector<int> &digit,
               vector<vector<float> > &tran,vector<vector<float> > &words);

vector<int> SEGmakeBoundaryOfT(vector<int> digit);

#endif /* defined(__speech_test__dtw__) */
