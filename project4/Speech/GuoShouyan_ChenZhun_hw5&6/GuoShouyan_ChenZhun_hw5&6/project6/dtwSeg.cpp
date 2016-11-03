//
//  dtwSeg.cpp
//  
//
//  Created by shouyanguo on 10/6/15.
//  Copyright (c) 2015 郭首彦. All rights reserved.
//


#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>
#include <float.h>

#include "dtwSeg.h"


using namespace std;


void SEGprintResult(int digit, vector<vector<float> > meanP,
	vector<vector<float> > variP,vector<vector<float> > tranP)
{
	//string dir = "C:/Users/june/Desktop/project7/train_result/resultB/";
	string dir = "C:/Users/june/Desktop/hwdata_assign6/isolate_train_result/";

	//const char* drinum[] = { "0.txt", "1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt", "8.txt", "9.txt" };
	const char* drinum[] = { "Z.txt", "1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt", "8.txt", "9.txt","silence.txt","O.txt" };

	ofstream outfile(dir + "mean" + drinum[digit]);

	for (int i = 0; i < meanP.size(); i++)
	{
		for (int j = 0; j < 39; j++)
		{
			outfile << meanP[i][j] << " ";
		}
		outfile << endl;
	}
	outfile.close();

	ofstream outfile1(dir + "vari" + drinum[digit]);
	for (int i = 0; i < variP.size(); i++)
	{
		for (int j = 0; j < 39; j++)
		{
			outfile1 << variP[i][j] << " ";
		}
		outfile1 << endl;
	}
	outfile1.close();

	ofstream outfile2(dir + "tran" + drinum[digit]);
	for (int i = 0; i < tranP.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			outfile2 << tranP[i][j] << " ";
		}
		outfile2 << endl;
	}
	outfile2.close();
}

// consider 0~9+s,silence(10)
// just read in the file, No prefixed dummy state
vector<vector<vector<float> > >  SEGreadTemplet(vector<string> &fileName)
{
	vector<vector<vector<float> > > templets(12);
	for(int i=0;i<12;i++)
	{
		//cout<<i<<endl;
		ifstream infile(fileName[i]);
		string line;
		vector<vector<float> > temp1;
		while(getline(infile, line))
		{
			vector<float> temp2;
			istringstream is(line);
			//char *word;
			string word;
			while(is>>word)
			{
				temp2.push_back(atof(word.data()));
			}
			temp1.push_back(temp2);
			//cout<<"size2="<<temp2.size()<<endl;
		}
		infile.close();
		templets[i]=temp1;
		//cout<<"size1="<<temp1.size()<<endl;
	}
	return templets;
}

// consider 0~9+s,silence(10)
// just read in the file, No prefixed dummy state
vector<vector<vector<float> > >  SEGreadVari(vector<string> &fileName)
{
	vector<vector<vector<float> > > templets(12);
	for(int i=0;i<12;i++)
	{
		//cout<<i<<endl;
		ifstream infile(fileName[i]);
		string line;
		vector<vector<float> > temp1;
		while(getline(infile, line))
		{
			vector<float> temp2;
			istringstream is(line);
			//char *word;
			string word;
			while(is>>word)
			{
				temp2.push_back(atof(word.data()));
			}
			temp1.push_back(temp2);
			//cout<<"size2="<<temp2.size()<<endl;
		}
		infile.close();
		templets[i]=temp1;
		//cout<<"size1="<<temp1.size()<<endl;
	}
	return templets;
}

// consider 0~9+s,silence(10)
// just read in the file, No prefixed dummy state
vector<vector<vector<float> > >  SEGreadTran( vector<string> &fileName)
{
	vector<vector<vector<float> > > templets(12);
	for(int i=0;i<12;i++)
	{
		//cout<<i<<endl;
		ifstream infile(fileName[i]);
		string line;
		vector<vector<float> > temp1;
		while(getline(infile, line))
		{
			vector<float> temp2;
			istringstream is(line);
			string word;
			while(is>>word)
			{
				temp2.push_back(atof(word.data()));
			}
			temp1.push_back(temp2);
			//cout<<"size2="<<temp2.size()<<endl;
		}
		infile.close();
		templets[i]=temp1;
		//cout<<"size1="<<temp1.size()<<endl;
	}
	return templets;
}

vector<vector<float> > SEGmakeWord(vector<vector<float> > &mfcc)
{
	vector<vector<float> > word;
	vector<float> temp(39,0);
	word.push_back(temp);
	for(int i=0;i<mfcc.size();i++)
		word.push_back(mfcc[i]);

	return word;
}

// concatenate (trained) mean/vari/tran according to the digit sequence 
// input: mean is the raw trained templets for every digits
// digit is the ground truth sequence of input digit
vector<vector<float> >  SEGmakeTemplet(vector<vector<vector<float> > > mean, vector<int> digit )
{
	vector<vector<float> > temp;
	vector<float> temp3(39);
	temp.push_back(temp3);
	for(int i=0; i<digit.size(); i++)
	{
		for(int j=0; j<mean[digit[i]].size(); j++)
		{
			temp.push_back(mean[digit[i]][j]);
		}
	}
	return temp;
}

// input: tran is the raw trained templets for every digits
// digit is the ground truth sequence of input digit
vector<vector<float> >  SEGmakeTran(vector<vector<vector<float> > > tran, vector<int> digit )
{
	vector<vector<float> > temp;

	for(int j=0; j<tran[digit[0]].size()-1; j++)
	{
		temp.push_back(tran[digit[0]][j]);
	}
	// modify the last state of tran
	vector<float> lastState = tran[digit[0]][tran[digit[0]].size()-1];
	lastState[1] = tran[digit[0]][tran[digit[0]].size()-2][1];  //modify, so that current digit can go to next digit(diagonal)
	temp.push_back(lastState);
	
	for(int i=1; i<digit.size(); i++)
	{
		for(int j=1; j<tran[digit[i]].size()-1; j++)  //starting from 1th
			temp.push_back(tran[digit[i]][j]);

		// modify the last state of tran
		vector<float> lastState = tran[digit[i]][tran[digit[i]].size()-1];
		lastState[1] = tran[digit[i]][tran[digit[i]].size()-2][1];  //modify, so that current digit can go to next digit(diagonal)
		temp.push_back(lastState);
	}
	return temp;
}



void SEGtraceBack(vector<vector<float> > &trellis,
               vector<vector<vector<vector<float> > > > &segmented,
               vector<int> &boundaryOfT, vector<int> &digit,
               vector<vector<float> > &tran,vector<vector<float> > &words)
{
    int x=(int)trellis.size()-1; //word
    int y=(int)trellis.back().size()-1; //templet
    int endX=x;
    int siteOfDIgit=digit.size()-1;
    float comp[3];
    while(x!=0 && y!=0)
    {
        comp[0]=trellis[x-1][y]+tran[y][0];
        comp[1]=trellis[x-1][y-1]+tran[y-1][1];
        if(y-2<0)
            comp[2]=FLT_MAX;
        else
            comp[2]=trellis[x-1][y-2]+tran[y-2][2];
        int min=(comp[0]>comp[1])?1:0;
        min=(comp[2]>comp[min])?min:2;
        if(min==0) //left
        {
            x=x-1;
        }
        else if(min==1) //diagonal
        {
            x=x-1;
            y=y-1;
        }
        else if(min==2)
            //else //sdiagonal
        {
            x=x-1;
            y=y-2;
        }
        else
		{
            cout<<"error"<<endl;
		}

        //check if pass one word
        if(y<=boundaryOfT[siteOfDIgit])
        {
            int startX=x+1;
			vector<vector<float> > temp;  // store one segment of digit
			while(startX<=endX)
            {
                temp.push_back(words[startX]);
				startX++;
            }
            segmented[digit[siteOfDIgit]].push_back(temp);
            //set to the next digit segment process
            siteOfDIgit=siteOfDIgit-1;
            endX=x;
			if (siteOfDIgit==0)
				break;
        }
    }
    //load in the last digit
    int startX=0;
	vector<vector<float> > temp;
	while(startX<=endX)
	{
		temp.push_back(words[startX]);
		startX++;
    }
    segmented[digit[siteOfDIgit]].push_back(temp);
}

vector<vector<float> > SEGmakeTrellis(vector<vector<float> >  templet, long wordLength)
{
    vector<vector<float> > score(wordLength);
    for(int i=0;i<wordLength;i++)
    {
        score[i].resize(templet.size());
    }
    
    for(int i=0;i<wordLength;i++)
    {
        score[i][0]=FLT_MAX;
    }
    for (int j = 0; j<templet.size(); j++)
    {
        score[0][j] = FLT_MAX;
    }
    score[0][0] = 0;
    return score;
}

float SEGdistance(vector<float> templet, vector<float> word, vector<float> vari)
{
    float sum1=0,sum2=0;
    float temp;
    for (int i=0;i<templet.size(); i++) {
        
        //cout << "i=" << i << endl;
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

float SEGgetMin(vector<vector<float> > &trellis,int i,int j,vector<vector<float> > tran)
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


void SEG_DP(vector<vector<float> >  &templet, vector<vector<float> > &words, vector<vector<float> > &trellis, vector<vector<float> > &vari, vector<vector<float> > &tran)
{
    for(int i=1;i<words.size();i++) // i for length of word
    {
        for(int j=1;j<templet.size();j++) //j for length of dict                 ////
        {
            //cout << i << " " << j <<" ";
            float min=SEGgetMin(trellis,i,j,tran);
            //cout << "min=" << min << " ";
            float d = SEGdistance(templet[j], words[i], vari[j]);
            //cout << i << " " << j << " ";
            trellis[i][j]=min+d;
        }   
    }
}

// make initial boundary of templets according to the input digit sequence
// MEMO: boundary will be modified during traceback (function: SEGtraceBack)
vector<int> SEGmakeBoundaryOfT(vector<int> digit)
{
	int site=1;
	vector<int> boundary;
	boundary.push_back(0);
	for(int i=0; i<digit.size();i++)
	{
		if(digit[i]!=10)    // 10 represent silence (2HMM), otherwise represent digit(5HMM).
			site=site+5;
		else
			site=site+2;
		boundary.push_back(site);
	}
	return boundary;
}