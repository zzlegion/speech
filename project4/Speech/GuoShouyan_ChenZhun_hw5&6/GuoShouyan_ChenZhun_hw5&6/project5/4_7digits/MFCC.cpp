//
//  MFCC.cpp
//  speech_test
//
//  Created by shouyanguo on 9/9/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
// 补零，改进三角滤波，DCT／4，归一化，合并到主函数
#include <iostream>
#include <fstream>
#include "MFCC.h"
#include <math.h>
#include <vector>
#include "fftw3.h"
using namespace std;

#define PI 3.14159
#define WL 1024 //window length, 44100*0.025=1102.5
#define numMel 40
#define sampleRate 44100
#define STEP 362 //1024-44100*0.015=1024-662=362

vector<float*> MFCC(short *data, int startPoint, int endPoint)
{
    //output file
    /*
    ofstream outfile;
    outfile.open("/Users/guo/Desktop/magnitude.txt");
    outfile.close();
    outfile.open("/Users/guo/Desktop/melResult.txt");
    outfile.close();
    outfile.open("/Users/guo/Desktop/iDct.txt");
    outfile.close();
    */
    // high pass filterring
    int dataL=endPoint-startPoint+1;
    float preemp[dataL];
    preemp[0]=data[startPoint];
    for(int i=1; i<dataL; i++)
    {
        preemp[i] = data[startPoint+i]-0.95*data[startPoint+i-1];
    }
    
    // use hamming window
    float w[WL];
    for(int n=0;n<WL;n++)
    {
        w[n] = 0.54-0.46*cos(2*PI*n/WL);
    }
    
    int sampleB=0; //start of the sample data
    vector<float*> cepstrums;
    //fftw3 parameters
    fftw_complex *in, *out;
    fftw_plan p,pDct;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WL);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * WL);
    
    //mel function
    // (num+1)*width=WL,num=40, set origin height=1
    float maxFre=1127*log(1+(float)sampleRate/2/700);
    float width=maxFre/(numMel+1);
    float freq2num=WL/(float)sampleRate;
    int mel[numMel][WL/2+1]={0};

    for(int i=0;i<numMel;i++)
    {
        int start=(int)(700 * (exp(i*width/1127)-1)*freq2num);
        int end=(int)(700 * (exp((i+2)*width/1127)-1)*freq2num);
        int mid=(end-start)/2+start;
        float h=width/(mid-start); //h=width/mid, k=h/mid
        float k1=h/(mid-start);
        float k2=h/(end-mid);

        for(int j=start;j<=end;j++)
        {
            if(j<mid)
                mel[i][j]=k1*(j-start);
            else
                mel[i][j]=k2*(end-j);
        }

        //mel[i][WL/2+1]=mid; //store every mid of the filter
    }
 
    while(sampleB<dataL)
    {
        //add window
        float window[WL]={0};
        int l=(sampleB+WL>dataL)?dataL-sampleB:WL;
        
        for(int i=0;i<l;i++)
        {
            window[i]=preemp[sampleB+i]*w[i];
        }
        // FFT
        for(int i=0;i<WL;i++)
        {
            in[i][0] = window[i]; //real
            in[i][1] = 0.0;    //image
        }
        p = fftw_plan_dft_1d(WL, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        
        double magnitude[WL/2+1]={0};
        for(int i=0;i<WL/2+1;i++)  //just need half of it
        {
            magnitude[i]=sqrt(out[i][0]*out[i][0]+out[i][1]*out[i][1]);
        }
        //...............write result...............
        /*
        outfile.open("/Users/guo/Desktop/magnitude.txt",ios::app);
        for(int j=0;j<256;j++)
        {
            outfile<<magnitude[j]<<" ";
        }
        outfile<<std::endl;
        outfile.close();
         */
        //...................................
        
        // mel function, set to 64
        double melResult[64]={0};
        float temp;
        for(int i=0;i<numMel;i++)
        {
            temp=0;
            for(int j=0;j<WL/2+1;j++)
            {
                temp=mel[i][j]*magnitude[j]+temp;
            }
            //log mel
            melResult[i]=log(temp);
        }
        
        //...............write result...............
        /*
        outfile.open("/Users/guo/Desktop/melResult.txt",ios::app);
        for(int j=0;j<numMel;j++)
        {
            outfile<<melResult[j]<<" ";
        }
        outfile<<std::endl;
        outfile.close();
         */
        //...................................

        // DCT,set to 64
        double melOut[64]={0};
        pDct = fftw_plan_r2r_1d(64,melResult,melOut,FFTW_REDFT10,FFTW_ESTIMATE);
        fftw_execute(pDct);
        float *cepstrum=new float[39];
        for(int i=0;i<39;i++)  //just need half of it
        {
            if(i<13)
                cepstrum[i]=melOut[i];
            else
                cepstrum[i]=0;
        }
        
        cepstrums.push_back(cepstrum);
        sampleB=sampleB+STEP;
         /*
        // iDCT,set to 64
        double idct[64]={0};
        pDct = fftw_plan_r2r_1d(64,melOut,idct,FFTW_REDFT01,FFTW_ESTIMATE);
        fftw_execute(pDct);
        //...............write result...............
       
        outfile.open("/Users/guo/Desktop/iDct.txt",ios::app);
        for(int j=0;j<numMel;j++)
        {
            outfile<<idct[j]<<" ";
        }
        outfile<<std::endl;
        outfile.close();
         */
        //...................................

    }
    fftw_destroy_plan(p);
    fftw_destroy_plan(pDct);
    fftw_free(in); fftw_free(out);
    
    //delta
    for(int i=1;i<cepstrums.size()-1;i++)
    {
        for(int j=0;j<13;j++)
        {
            cepstrums[i][j+13]=cepstrums[i+1][j]-cepstrums[i-1][j];
        }
    }
    //two delta
    for(int i=2;i<cepstrums.size()-2;i++)
    {
        for(int j=0;j<13;j++)
        {
            cepstrums[i][j+26]=cepstrums[i+1][j+13]-cepstrums[i-1][j+13];
        }
    }
    for(int j=0;j<39;j++)
    {
        cepstrums[0][j]=0;
    }
    vector<float*>::iterator b=cepstrums.begin();
    b=cepstrums.erase(b);
    cepstrums.erase(b);
    cepstrums.pop_back();
    cepstrums.pop_back();
    //normalized
    float mean[39]={0};
    float variation[39]={0};
    float temp;
    for(int i=0;i<cepstrums.size();i++)
    {
        for(int j=0;j<39;j++)
        {
            mean[j]+=cepstrums[i][j];
        }
    }
    for(int j=0;j<39;j++)
    {
        mean[j]=mean[j]/cepstrums.size();
    }
    for(int i=0;i<cepstrums.size();i++)
    {
        for(int j=0;j<39;j++)
        {
            temp=cepstrums[i][j]-mean[j];
            cepstrums[i][j]=temp;
            variation[j]+=temp*temp;
        }
    }
    //variation
    for(int j=0;j<39;j++)
    {
        variation[j]=sqrt(variation[j]/cepstrums.size());
    }
    for(int i=0;i<cepstrums.size();i++)
    {
        for(int j=0;j<39;j++)
        {
            cepstrums[i][j]=cepstrums[i][j]/variation[j];
        }
    }
    
    return cepstrums;
}