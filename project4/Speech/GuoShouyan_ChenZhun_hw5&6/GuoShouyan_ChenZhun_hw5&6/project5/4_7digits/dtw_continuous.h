#ifndef __DTW__continuous__
#define __DTW__continuous__

#include <stdio.h>
#include <vector>
using namespace std;

void makeMean(const char* Mean[], vector<vector<float> > &silenceMean,vector<vector<vector<float> > > &digitMean );
void makeVari(const char* Vari[], vector<vector<float> > &silenceVari,vector<vector<vector<float> > > &digitVari );
void makeTran(const char* Tran[], vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran );
vector<vector<vector<vector<float> > > >  makeTemplet(const char *Mean[]);
vector<vector<float> > makeWord(vector<float*> &mfcc);
vector<vector<vector<vector<float> > > > makeTrellis( vector<vector<vector<vector<float> > > > &templets, vector<vector<float> > &words);
float distance(vector<float> templet, vector<float> word, vector<float> vari);
float getMin(vector<vector<vector<vector<float> > > > &trellis,int i, int j, int k, int n, vector<vector<float> > &Tran);
vector<int> traceback(vector<vector<vector<vector<float> > > > &trellis,vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran,float &cost);
float DP (vector<vector<vector<vector<float> > > > &trellis,
          vector<vector<vector<vector<float> > > > &templets, vector<vector<float> > &words,
          vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran,
          vector<vector<float> > &silenceVari,vector<vector<vector<float> > > &digitVari);
// real time test
vector<int> dtw(const char *Mean[],const char *Vari[],const char *Tran[],vector<float*> &mfcc);
// for test
//vector<int> dtw(const char *Mean[],const char *Vari[],const char *Tran[],vector<vector<float> > &mfcc);
#endif