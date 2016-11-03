//
//  segment.cpp
//  project7
//
//  Created by shouyanguo on 11/27/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//
///////////////////////////////////training continuous model//////////////////////////////
#include <fstream>
#include <sstream>
#include <iostream>
#include "segment.h"
#include "dtwSeg.h"
#include "train1.h"
#include "getFile.h"

// just record the 0~9+s, consider silence, zero has 2 pronounciations
vector<vector<vector<float> > > MEAN ;  // 1D is 0~9+s, 2D is 5 segments, 3D is 39 dim
vector<vector<vector<float> > > VARI ;  // 1D is 0~9+s, 2D is 5 segments, 3D is 39 dim
vector<vector<vector<float> > > TRAN ;  // 1D is 0~9+s, 2D is 5 segments, 3D is 3 direction


void getIsolateDataFileName(vector<string>& MeanFile, vector<string> &VariFile, vector<string> &TranFile)
{
	

	char * IsolateMeanFile = "C:\\Users\\june\\Desktop\\hwdata_assign6\\isolate_train_result\\mean";
	char * IsolateVariFile = "C:\\Users\\june\\Desktop\\hwdata_assign6\\isolate_train_result\\vari";
	char * IsolateTranFile = "C:\\Users\\june\\Desktop\\hwdata_assign6\\isolate_train_result\\tran";
	


	getFiles(IsolateMeanFile, "txt", MeanFile); 
	getFiles(IsolateVariFile, "txt", VariFile); 
	getFiles(IsolateTranFile, "txt", TranFile); 
}

vector<vector<int> > SEGcalSegLength(vector<vector<vector<vector<float> > > > &segmented)
{
	vector<vector<int> > temp(segmented.size());
	for (int i = 0; i < temp.size(); i++)
	{
		temp[i].resize(segmented[i].size());
	}


	for (int i = 0; i < segmented.size(); i++)
	{
		for (int j = 0; j < segmented[i].size(); j++)
		{
			temp[i][j] = segmented[i][j].size();
			cout << temp[i][j] << " ";
		}
	}
	return temp;
}

float SEGcalSegChange(vector<vector<int> > &preSegLength, vector<vector<int> > &curSegLength)
{
	float change=0;
	int num = 0;
	for (int i = 0; i < preSegLength.size(); i++)
	{
		for (int j = 0; j < preSegLength[i].size(); j++)
		{
			num++;
			change += abs(preSegLength[i][j] - curSegLength[i][j]);
		}
	}
	change = change / num;
	return change;
}

// mfcc is the total input of audio data, digit is the corresponding ground truth digit sequence
void segment(vector<vector<vector<float> > > mfcc, vector<vector<int> > digit)
{
    //1D 0~9+silence digit, 2D num of sample of one digit, 3D num of mfcc state of one sample, 4D of 39
	// to store the segments of samples in bins(0~9+silence)
	

	
	// read in isolated trained mfcc of 0~9, don't include silence
	// initialization
	vector<string> MeanFileName;
	vector<string> VariFileName;
	vector<string> TranFileName;
	getIsolateDataFileName(MeanFileName,VariFileName,TranFileName);

	// MEAN,VARI,TRAN consider silence
	MEAN = SEGreadTemplet(MeanFileName);
	VARI = SEGreadVari(VariFileName);
	TRAN = SEGreadTran(TranFileName);

	int count=0;  //iterate times

	float change = FLT_MAX;
	vector<vector<int> > preSegLength;
	vector<vector<int> > curSegLength;


	while(change > 2)
	{

		//1D 0~9+silence digit, 2D num of sample of one digit, 3D num of mfcc state of one sample, 4D of 39
		// to store the segments of samples in bins(0~9+silence)
		//vector<vector<vector<vector<float> > > > segmented(11);   //////////////////////////////////guo

		// haven't consider zero as "O"
		//vector<vector<vector<vector<float> > > > segmented(11);

		// consider zero as "O"
		vector<vector<vector<vector<float> > > > segmented(12);



		//do segmenting
		count++;

        for(int i=0; i<mfcc.size();i++)
        {
			vector<vector<float> > word=SEGmakeWord(mfcc[i]);       

            vector<vector<float> > templet=SEGmakeTemplet(MEAN,digit[i]);

            vector<vector<float> > vari=SEGmakeTemplet(VARI,digit[i]);
  
            vector<vector<float> > tran=SEGmakeTran(TRAN,digit[i]);	   ////////
        
            vector<vector<float> > trellis=SEGmakeTrellis(templet, word.size());
            
			SEG_DP(templet,word,trellis,vari,tran);
        
            vector<int> boundaryOfT=SEGmakeBoundaryOfT(digit[i]);

			/*
			for(int j=0;j<digit[i].size();j++)
				cout<<digit[i][j]<<" ";
			cout<<endl;
			cout<<digit[i].size()<<endl;
            */
			SEGtraceBack(trellis, segmented, boundaryOfT, digit[i], tran, word);
        }

		//calculate the change of segment
		curSegLength = SEGcalSegLength(segmented);

		if (count == 1)
			change = FLT_MAX;
		else
			change = SEGcalSegChange(preSegLength, curSegLength);

		preSegLength = curSegLength;

		//start training for every digit
		//update MEAN VARI TRAN
		for(int j=0;j<12;j++)
		{
			train1(segmented[j],j);
			if(j==10)
				cout<<"finish training silence"<<endl;
			else cout<<"finish digit"<<j<<endl;
		}

		cout << "iterate time: "<< count << "   change:" << change << endl;

	}

	for(int i=0;i<12;i++)  ////////////////////////guo
	{
		SEGprintResult(i, MEAN[i],VARI[i],TRAN[i]);
	}
}