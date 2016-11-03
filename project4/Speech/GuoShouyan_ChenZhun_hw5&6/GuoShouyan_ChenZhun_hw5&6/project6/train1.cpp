//
//  train1.cpp
//  Train
//
//  Created by shouyanguo on 10/8/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//

#include "train1.h"
#include <vector>
#include "dtw.h"
#include <iostream>
#include <math.h>
#include <float.h>
#include <fstream>
#include <string>
using namespace std;

// all digits
extern vector<vector<vector<float> > > MEAN ;
extern vector<vector<vector<float> > > VARI ;
extern vector<vector<vector<float> > > TRAN ;


// single digit
vector<vector<float> > mean;
vector<vector<float> > vari;
vector<vector<float> > tran;  //consider T0j


int SEGMENT;         // num of HMM SEGMENT: digit(5), silence(2)

vector<int> segStateNum;

void calMean(vector<vector<vector<float> > > &sample, vector<vector<int> > &boundary)
{
    for (int i = 0; i < mean.size(); i++)
    {
        for (int j = 0; j < 39; j++)
        {
            mean[i][j] = 0;
        }
    }
    
    for(int i=0;i<sample.size();i++)
    {
        for(int j=0;j<SEGMENT;j++)
        {
            for(int k=boundary[i][j];k<boundary[i][j+1];k++)
            {
                if(k<boundary[i][j+1]) //check whether segment exit
                {
                    for(int l=0;l<39;l++)
                    {
                        mean[j][l]+=sample[i][k][l]/segStateNum[j];
                    }
                }
            }
        }
    }
}

void calVari(vector<vector<vector<float> > > &sample, vector<vector<int> > &boundary)
{
    for (int i = 0; i < vari.size(); i++)
    {
        for (int j = 0; j < 39; j++)
        {
            vari[i][j] = 0;
        }
    }
    
    float temp;
    for(int i=0;i<sample.size();i++)
    {
        for(int j=0;j<SEGMENT;j++)
        {
            for(int k=boundary[i][j];k<boundary[i][j+1];k++)
            {
                if(k<boundary[i][j+1])
                {
                    for(int l=0;l<39;l++)
                    {
                        temp=sample[i][k][l]-mean[j][l];
                        vari[j][l]+=temp*temp/segStateNum[j];
                    }
                }
            }
        }
    }
}

void calTran(vector<vector<int> > &boundary)
{
    
    vector<float> T0j(3);
    //vector<int> startSegIndex(SEGMENT); // histogram to record the start segment index for each sample
    int start = 0;
    for (int i = 0; i < boundary.size(); i++)
    {
        if (boundary[i][0] == boundary[i][1])
            start++;
    }
    T0j[0] = FLT_MAX;
    T0j[1] = -log(((float)(boundary.size()-start) / boundary.size()));
    if (start == 0)
        T0j[2] = FLT_MAX;
    else
        T0j[2] = -log(((float)(start) / boundary.size()));
    /*
     for (int i=0;i<boundary.size();i++)
     {
     if (startSegIndex[i]!=0)
     T0j[i] = -log((float)(startSegIndex[i]/boundary.size()));
     else
     T0j[i] = FLT_MAX;
     }
     */
    tran[0] = T0j;
    
    for (int i =0; i<SEGMENT-1; i++)
    {
        if (segStateNum[i] == boundary.size())
            tran[i + 1][0] = FLT_MAX;
        else
            tran[i+1][0]=-log((float)(segStateNum[i]-boundary.size())/(float)segStateNum[i]);
    }
    int sum;
    for(int i=0;i<SEGMENT-1;i++)
    {
        sum=0;
        for(int j=0;j<boundary.size();j++)
        {
            if(boundary[j][i]==boundary[j][i+1])
                sum++;
        }
        tran[i+1][1]=-log((float)(boundary.size()-sum)/(float)segStateNum[i]);
        if(sum==0)
            tran[i+1][2]=FLT_MAX;
        else
            tran[i+1][2]=-log((float)sum/(float)segStateNum[i]);
    }
}

float calBoundary(vector<vector<int> > &boundary,vector<vector<vector<float> > > &sample,vector<float> &errorRate)
{
    float errorSum=0;
    vector<vector<float> >  templet=makeTemplet(mean);
    
    for(int i=0;i<sample.size();i++)
    {
        vector<vector<float> > word= makeWord(sample[i]);
        vector<vector<float> > trellis = makeTrellis(templet, word.size());
        float score=DP(templet,word,trellis,vari,boundary[i],tran);
        //cout << "score="<<score <<endl;
        errorSum+=fabs(score-errorRate[i]);
        //cout << errorRate[i] << endl;
        errorRate[i]=score;
    }
    
    return errorSum;
}

void calSegStateNum(vector<vector<int> > &boundary)
{
    for(int i=0;i<boundary.size();i++)
    {
        for(int j=0;j<boundary[i].size()-1;j++)
        {
            if(i==0)
                segStateNum[j]=boundary[i][j+1]-boundary[i][j];   // initialize
            else
                segStateNum[j]+=boundary[i][j+1]-boundary[i][j];
        }
    }
}

void printResult(int digit)
{
    string dir = "/Users/guo/Desktop/result/";
    string drinum[] = { "Z.txt", "1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt", "8.txt", "9.txt","silence.txt","O.txt" };
    ofstream outfile(dir + "mean" + drinum[digit]);
    for (int i = 0; i < mean.size(); i++)
    {
        for (int j = 0; j < 39; j++)
        {
            outfile << mean[i][j] << " ";
        }
        outfile << endl;
    }
    outfile.close();
    
    ofstream outfile1(dir + "vari" + drinum[digit]);
    for (int i = 0; i < vari.size(); i++)
    {
        for (int j = 0; j < 39; j++)
        {
            outfile1 << vari[i][j] << " ";
        }
        outfile1 << endl;
    }
    outfile1.close();
    
    ofstream outfile2(dir + "tran" + drinum[digit]);
    for (int i = 0; i < tran.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            outfile2 << tran[i][j] << " ";
        }
        outfile2 << endl;
    }
    outfile2.close();
}

void train1(vector<vector<vector<float> > > &sample, int digit)
{
	cout<<digit<<endl;

    // decide if silence(10) or 0~9,
    if(digit==10)
        SEGMENT=2;
    else
        SEGMENT=5;
    
    //initial boundary
    //1d is num of sample of one digit, 2d is boundary for one digit
    //boundary[0][0] = start point of first segment of first sample
    //boundary[][i]==boundary[][i+1] means there is no segment i
    vector<float> errorRate(sample.size(),0);
    vector<vector<int> > boundary(sample.size());

    //initial mean,vari,tran
    mean=MEAN[digit];
    vari=VARI[digit];
    tran=TRAN[digit];
    
    float errorSum=1000;
    //initial boundary and seg state num
    for(int i=0;i<sample.size();i++)
    {
        vector<int> b2(SEGMENT+1,0);
        boundary[i]=b2; 
    }
    errorSum=calBoundary(boundary,sample,errorRate);

    segStateNum.resize(SEGMENT);
    
    int time = 0;

    while (errorSum>5) {
        calSegStateNum(boundary);
        calMean(sample,boundary);
        calVari(sample,boundary);
        calTran(boundary);
        
		//update boundary
        errorSum=calBoundary(boundary,sample,errorRate);
        //cout << "iteration time=" << time << ", errorSum=" << errorSum << endl;
        time++;
    }
    
    //initial mean,vari,tran
    MEAN[digit]=mean;
    VARI[digit]=vari;
    TRAN[digit]=tran;
}