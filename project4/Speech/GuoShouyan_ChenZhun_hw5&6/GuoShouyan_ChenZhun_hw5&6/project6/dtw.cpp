//
//  dtw.cpp
//  speech_test
//
//  Created by shouyanguo on 10/6/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//

#include "dtw.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>
#include <float.h>
using namespace std;

vector<vector<float> >  makeTemplet(vector<vector<float> > &mean)
{
    vector<vector<float> > templets(mean.size()+1);                         ///
    vector<float> temp(39,0);   //dummy
    templets[0] =temp;
    for(int i=0;i<mean.size();i++)
    {
        templets[i+1]=mean[i];
        //cout<<"size1="<<temp1.size()<<endl;
    }
    return templets;
}

vector<vector<float> > makeWord(vector<vector<float> > mfcc)
{
    vector<vector<float> > words;
    //add prefix
    vector<float> temp3(39);
    words.push_back(temp3);
    vector<float> word(39);
    for(int i=0;i<mfcc.size();i++)
    {
        for(int j=0;j<39;j++)
        {
            word[j]=mfcc[i][j];
        }
        words.push_back(word);
    }
    return words;
}

void printMatrix(vector<vector<float> > score)
{
    for(int i=(int)score[0].size()-1;i>-1;i--)
    {
        for(int j=0;j<score.size();j++)
        {
            cout<<score[j][i]<<" ";
        }
        cout<<endl;
    }
}

void traceBack(vector<vector<float> > &trellis, vector<int> &boundary,vector<vector<float> > &tran)
{
    int x= trellis.size()-1;		//word
    int y= trellis.back().size()-1; //templet

	boundary[y] = x;    /////////// ONLY for proj7

    float comp[3];

	while(x != 0 && y != 0)
    {
		if(x<0||y<0)
		{
			cout<<" x<0 or y<0"<<endl;
			break;
		}
        comp[0]=trellis[x-1][y]+tran[y][0];
        comp[1]=trellis[x-1][y-1]+tran[y-1][1];
        if(y-2<0)
            comp[2]=FLT_MAX;
        else
            comp[2]=trellis[x-1][y-2]+tran[y-2][2];

		float minvalue = comp[0]<comp[1]?comp[0]:comp[1];
		minvalue = minvalue<comp[2]?minvalue:comp[2];
		if(minvalue==FLT_MAX)
		{
			cout<<"error"<<endl;
			cout<<"x:"<<x<<" y:"<<y<<endl;
			system("pause");
		}

        int min=(comp[0]>comp[1])?1:0;
        min=(comp[2]>comp[min])?min:2;
		if(min==0) //left
        {
            x=x-1;
        }
        else if(min==1) //diagonal
        {
            boundary[y-1]=x-1;
            x=x-1;
            y=y-1;
        }
        else if(min==2)
        //else //sdiagonal
        {
            boundary[y-1]=x-1;
            boundary[y-2]=x-1;
            x=x-1;
            y=y-2;
        }
        else{
            cout<<"error"<<endl;
        }
        
    }
    //boundary[0]=0;
}

vector<vector<float> > makeTrellis(vector<vector<float> >  templet, long wordLength)
{
    vector<vector<float> > score(wordLength);
    for(int i=0;i<wordLength;i++)
    {
        score[i].resize(templet.size());
    }
    
    for(int i=0;i<wordLength;i++)
    {
        score[i][0]=FLT_MAX;                          /////
    }
    for (int j = 0; j<templet.size(); j++)
    {
        score[0][j] = FLT_MAX;
    }
    score[0][0] = 0;
    return score;
}

float distance(vector<float> templet, vector<float> word, vector<float> vari)
{
    float sum1=0,sum2=0;
    float temp;
    for (int i=0;i<templet.size(); i++) {
        sum1 = sum1 + log(2 * 3.1416*vari[i]);
        //cout << "sum1=" <<sum1<< endl;
        temp=word[i]-templet[i];
        //cout << "sum2" << endl;
        //sum2 = sum2 + temp*temp / variSqare;
        sum2 = sum2 + temp*temp/vari[i];
        //cout << "sum2="<<sum2 << endl;
        //sum2+=fabs(word[i]-templet[i]);
    }
    return 0.5*sum1+0.5*sum2;
}

float getMin(vector<vector<float> > &trellis,int i,int j,vector<vector<float> > tran)
{
    float left=FLT_MAX,sdiagonal=FLT_MAX,diagonal=FLT_MAX;
    left=trellis[i-1][j]+tran[j][0];
    if(j-1>=0)
        diagonal=trellis[i-1][j-1]+tran[j-1][1];
    if(j-2>=0)
        sdiagonal=trellis[i-1][j-2]+tran[j-2][2];
    
    float min;
    min = (left > diagonal) ? diagonal : left;
    min=(sdiagonal>min)?min:sdiagonal;
    return min;
}

float DP(vector<vector<float> >  &templet, vector<vector<float> > &words, vector<vector<float> > &trellis, vector<vector<float> > &vari, vector<int> &boundary,vector<vector<float> > tran)
{
    for(int i=1;i<words.size();i++) // i for length of word
    {
        for(int j=1;j<templet.size();j++) //j for length of dict                 ////
        {
            //cout << i << " " << j <<" ";
            float min=getMin(trellis,i,j,tran);
            //cout << "min=" << min << " ";
            float d = distance(templet[j], words[i], vari[j-1]);
            //cout << i << " " << j << " ";
            trellis[i][j]=min+d;
            //cout << "distance="<<d << endl;
        }
        
    }
    //cout<<"trellis"<<endl;
    //printMatrix(trellis);
    traceBack(trellis,boundary,tran);
    return trellis[words.size()-1][templet.size()-1];
}
