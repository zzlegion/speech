
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>
#include <float.h>
#include "dtw_continuous.h"

#define  PROB_SILENCE (0.1)
#define  PRUNE (1)

using namespace std;

void makeVari(const char* Vari[], vector<vector<float> > &silenceVari,vector<vector<vector<float> > > &digitVari )     // idx=10 : dummy  idx=0~9: digit0~9
{
    // silenceVari
    for(int i=0;i<3;i++)
        silenceVari[i].resize(39);
    
    vector<float> temp(39,0);
    silenceVari[0] = temp;
    
    ifstream infile(Vari[10]);   // silence
    
    string line;
    for(int j=0;j<2;j++)
    {
        getline(infile,line);
        vector<float> temp;
        istringstream is(line);
        string word;
        while(is>>word)
        {
            temp.push_back(atof(word.data()));
        }
        silenceVari[j+1] = temp;
    }
    
    // digitVari
    for(int i=0;i<10;i++)
    {
        digitVari[i].resize(6);
        for(int j=0;j<6;j++)
            digitVari[i][j].resize(39);
    }
    vector<float> dummy(39,0);
    for(int idx=0;idx<10;idx++)  //0~9
    {
        digitVari[idx][0] = dummy;
        ifstream infile(Vari[idx]);
        string line;
        for(int j=0;j<5;j++)
        {
            getline(infile,line);
            vector<float> temp;
            istringstream is(line);
            string word;
            while(is>>word)
            {
                temp.push_back(atof(word.data()));
            }
            digitVari[idx][j+1] = temp;
        }
    }
}

void makeTran(const char* Tran[], vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran )     // idx=10 : dummy  idx=0~9: digit0~9
{
    // silenceTran
    for(int i=0;i<3;i++)
        silenceTran[i].resize(3);
    
    ifstream infile(Tran[10]);
    string line;
    for(int j=0;j<3;j++)
    {
        getline(infile,line);
        vector<float> temp;
        istringstream is(line);
        string word;
        while(is>>word)
        {
            temp.push_back(atof(word.data()));
        }
        silenceTran[j] = temp;
    }
    
    
    // digitTran
    for(int i=0;i<10;i++)
    {
        digitTran[i].resize(6);
        for(int j=0;j<6;j++)
            digitTran[i][j].resize(3);
    }
    for(int idx=0;idx<10;idx++)  //0~9
    {
        ifstream infile(Tran[idx]);
        string line;
        for(int j=0;j<6;j++)
        {
            getline(infile,line);
            vector<float> temp;
            istringstream is(line);
            string word;
            while(is>>word)
            {
                temp.push_back(atof(word.data()));
            }
            digitTran[idx][j] = temp;
        }
    }
}

// templet:
// 7 digitState
// 6 x digitStateTemplet1 (10digit) : 10x(5+2silence)x39
// 1 x digitStateTemplet2 (8 digit) : 8x(5+2silence)x39
// inside a State, 1 non-emittingState (x1) at the beginning, to store when
//				   following StateNum (x5)

#define STATE (7)			   // total State for TEL Digit
vector<vector<vector<vector<float> > > >  makeTemplet(const char *Mean[])
{
    // record mfcc data for silence
    vector<vector<float> > mfccSilence;
    ifstream infile(Mean[10]);
    string line;
    vector<float> temp(39,0);
    mfccSilence.push_back(temp);
    for(int j=0;j<2;j++)   // assume silence only have 2 HMM state
    {
        getline(infile,line);  // 1 row
        vector<float> temp3;   // store 39 dim mfcc of 1 state (1 row)
        istringstream is(line);
        string word;
        while(is>>word)
        {
            temp3.push_back((atof(word.data())));
        }
        mfccSilence.push_back(temp3);
    }
    
    // record all mfcc data for all digits
    // fileName[i] : i=0:9
    vector<vector<vector<float> > > mfccDigit(10);
    for(int i=0;i<10;i++)    // 0~9
    {
        ifstream infile(Mean[i]);
        string line;
        vector<vector<float> > oneDigit;  //6x39
        vector<float> temp(39,0);
        oneDigit.push_back(temp);
        while(getline(infile,line))
        {
            vector<float> temp3;   // store 39 dim mfcc of 1 state (1 row)
            istringstream is(line);
            string word;
            while(is>>word)
            {
                temp3.push_back((atof(word.data())));
            }
            oneDigit.push_back(temp3);
        }
        mfccDigit[i] = oneDigit;
    }
    infile.close();
    
    
    // [j] x [k] x [n] x [l]
    // totalStateNum x  digit(8/10) x (5dig+2sil+1dummy)HMMstates x 39
    vector<vector<vector<vector<float> > > > templets (STATE*2);      // silence state + TEL state
    
    //1.  Tel states
    for(int j=0;j<STATE;j++)
    {
        if(j==0)
        {
            vector<vector<vector<float> > > telState1(8);   // 8x(5+1)x39  (k n l)
            for (int k=0;k<8;k++)
            {
                telState1[k].resize(6);
                for (int n=0;n<6;n++)
                    telState1[k][n].resize(39);
            }
            for (int k=0;k<8;k++)
            {
                for(int n=0;n<6;n++)
                {
                    for(int l=0;l<39;l++)
                    {
                        telState1[k][n][l] = mfccDigit[k+2][n][l];
                    }
                }
            }
            templets[j*2+1] = telState1;
        }
        else
        {
            vector<vector<vector<float> > > telState(10);   // 10x6x39  (k n l)
            for (int k=0;k<10;k++)
            {
                telState[k].resize(6);
                for (int n=0;n<6;n++)
                    telState[k][n].resize(39);
            }
            for (int k=0;k<10;k++)
            {
                for(int n=0;n<6;n++)
                {
                    for(int l=0;l<39;l++)
                    {
                        telState[k][n][l] = mfccDigit[k][n][l];
                    }
                }
            }
            templets[j*2+1] = telState;
        }
    }
    
    //2. Silence State
    for(int j=0;j<STATE;j++)
    {
        vector<vector<vector<float> > > silenceState(1);   // 1x(2+1)x39  (k n l)
        for (int k=0;k<1;k++)
        {
            silenceState[k].resize(3);
            for (int n=0;n<3;n++)
                silenceState[k][n].resize(39);
        }
        for(int n=0;n<3;n++)
        {
            for(int l=0;l<39;l++)
            {
                silenceState[0][n][l] = mfccSilence[n][l];
            }
        }
        templets[j*2] = silenceState;
    }
    
    return templets;
    
}

// real time detect
//vector<vector<float> > makeWord(vector<float*> &mfcc)
// for debug
vector<vector<float> > makeWord(vector<float* > &mfcc)
{
    vector<vector<float> > words;
    //add prefix
    vector<float> temp3(39,0);
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

vector<vector<vector<vector<float> > > > makeTrellis( vector<vector<vector<vector<float> > > > &templets, vector<vector<float> > &words)
{
    //wordLength,telState(digit/silence),digitType,hmmState
    // i, j, k, n
    vector<vector<vector<vector<float> > > > trellis(words.size());
    
    for(int i=0;i<trellis.size();i++)
    {
        trellis[i].resize(templets.size());
        for(int j=0;j<templets.size();j++)  // telState
        {
            if(j%2==0) // silence state
            {
                trellis[i][j].resize(1);
                trellis[i][j][0].resize(3);  // 2hmm + 1dummy
                for(int n=0;n<3;n++)
                    trellis[i][j][0][n]=FLT_MAX;
            }
            else // telDigit state
            {
                if(j==1)  // the tel State without "0" and "1"
                {
                    trellis[i][j].resize(8);
                    for(int k=0;k<8;k++)
                    {
                        trellis[i][j][k].resize(6);
                        for(int n=0;n<6;n++)
                            trellis[i][j][k][n]=FLT_MAX;
                    }
                }
                else
                {
                    trellis[i][j].resize(10);
                    for(int k=0;k<10;k++)
                    {
                        trellis[i][j][k].resize(6);
                        for(int n=0;n<6;n++)
                            trellis[i][j][k][n]=FLT_MAX;
                    }
                }
            }
        }
    }
    //initial
    // the first column
    for(int j=0;j<STATE*2;j++)
    {
        for(int k=0;k<trellis[0][j].size();k++)
            for(int n=0;n<trellis[0][j][k].size();n++)  // n th hmm
                trellis[0][j][k][n]=FLT_MAX;
    }
    
    // the first element of the first column of TEL STATE 0(j=0,j=1) and 3(j=6,j=7)
    
    for(int j=1;j<=7;j=j+6)  //digit state
    {
        for(int k=0;k<trellis[0][j].size();k++)
        {
            trellis[0][j][k][0] = 0;
        }
    }
    for(int j=0;j<=6;j=j+6)  //silence state
        trellis[0][j][0][0] = 0;
    
    
    
    // the first row(dummy) for the first TEL STATE and 4th STATE
    for(int i=1;i<trellis.size();i++)
    {
        for(int j=0;j<=6;j=j+6)
            trellis[i][j][0][0] = FLT_MAX;
    }
    
    return trellis;
}


float distance(vector<float> templet, vector<float> word, vector<float> vari)
{
    float sum1=0,sum2=0;
    float temp;
    for (int i=0;i<templet.size(); i++) {
        sum1 = sum1 + log(2 * 3.1416*vari[i]);
        temp=word[i]-templet[i];
        sum2 = sum2 + temp*temp / vari[i];
    }
    float dis = (0.5*sum1+0.5*sum2);
    return dis;
}

// i column, j state(digit or silence state),  k(digit), n(hmm)
float getMin(vector<vector<vector<vector<float> > > > &trellis,int i, int j, int k, int n, vector<vector<float> > &Tran)
{
    float left=FLT_MAX,sdiagonal=FLT_MAX,diagonal=FLT_MAX;
    left = trellis[i-1][j][k][n]+Tran[n][0];
    if(n-1>=0)
        diagonal = trellis[i-1][j][k][n-1]+Tran[n-1][1];
    if(n-2>=0)
        sdiagonal = trellis[i-1][j][k][n-2]+Tran[n-2][2];
    float min;
    min = (left > diagonal) ? diagonal : left;
    min=(sdiagonal>min)?min:sdiagonal;
    return min;
}

vector<int> traceback(vector<vector<vector<vector<float> > > > &trellis,vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran,float &cost)
{
    vector<int> result;
    int x = trellis.size()-1;
    //while(x!=0)
    //{
    for(int j=STATE-1;j>=0;j--)
    {
        //cout<<j<<"STATE"<<endl;
        if(x==0)
            break;
        // find the digit index with min cost of a Digit STATE
        float min = FLT_MAX;
        int digitIdx;
        for(int k=0;k<trellis[x][j*2+1].size();k++)
        {
            int pos = trellis[x][j*2+1][k].size()-1;
            //cout<<"pos:"<<pos<<endl;
            
            if(trellis[x][j*2+1][k][pos]<min)
            {
                min = trellis[x][j*2+1][k][pos];
                digitIdx = k;
            }
        }
        if(x==trellis.size()-1)
            cost = min;
        if(j!=0)
            result.push_back(digitIdx);
        else
            result.push_back(digitIdx+2);
        
        // trace back inside the current digit STATE
        float comp[3];
        int y = trellis[x][j*2+1][digitIdx].size()-1;   // y=5
        
        
        while(y!=0 && x!=0)
        {
            comp[0]=trellis[x-1][j*2+1][digitIdx][y]+digitTran[digitIdx][y][0];
            comp[1]=trellis[x-1][j*2+1][digitIdx][y-1]+digitTran[digitIdx][y-1][1];
            if(y-2<0)
                comp[2]=FLT_MAX;
            else
                comp[2]=trellis[x-1][j*2+1][digitIdx][y-2]+digitTran[digitIdx][y-2][2];
            int min=(comp[0]>comp[1])?1:0;
            min=(comp[2]>comp[min])?min:2;
            
            //cout<<"x:"<<x<<"   y:"<< y <<"previous one's min: "<< endl;
            if(min==0) //left
            {
                x=x-1;
                //cout<<comp[0]<<endl;
            }
            else if(min==1) //diagonal
            {
                x=x-1;
                y=y-1;
                //cout<<comp[1]<<endl;
            }
            else if(min==2)	//sdiagonal
            {
                x=x-1;
                y=y-2;
                //cout<<comp[2]<<endl;
            }
        }
        
        //judge whether have passed Silence State or not
        
        if(trellis[x][j*2][0][0] > trellis[x][j*2][0][2])  //go passed silence///////////////////////
        {
            result.push_back(-1);  // -1 represents silence state
            //trace back in the silence state
            int y = trellis[x][j*2][0].size()-1;   // y=2
            while(y!=0 && x!=0)
            {
                comp[0]=trellis[x-1][j*2][0][y]+silenceTran[y][0];
                
                if(y!=1)
                    comp[1]=trellis[x-1][j*2][0][y-1]+silenceTran[y-1][1];
                
                else   // whether go back to silence state or not
                {
                    float temp = trellis[x-1][j*2][0][0]<trellis[x-1][j*2][0][2]?trellis[x-1][j*2][0][0]:trellis[x-1][j*2][0][2];
                    comp[1]=temp+silenceTran[y-1][1];
                }
                
                if(y-2<0)
                    comp[2]=FLT_MAX;
                else
                    comp[2]=trellis[x-1][j*2][0][y-2]+silenceTran[y-2][2];
                int min=(comp[0]>comp[1])?1:0;
                min=(comp[2]>comp[min])?min:2;
                if(min==0) //left
                {
                    x=x-1;
                }
                else if(min==1) //diagonal
                {
                    x=x-1;
                    if(j!=1)
                        y=y-1;
                    else       // whether go back to silence state or not
                    {
                        int flag = trellis[x-1][j*2][0][2]<trellis[x-1][j*2][0][0]?1:0;
                        if(flag==0)  // won't go back to silence state
                            y=y-1;
                        else
                            y=trellis[x][j*2][0].size()-1;
                    }
                }
                else if(min==2)	//sdiagonal
                {
                    x=x-1;
                    y=y-2;
                }
            }
        }
        
    }
    //}
    
    return result;
}

float DP (vector<vector<vector<vector<float> > > > &trellis,
          vector<vector<vector<vector<float> > > > &templets, vector<vector<float> > &words,
          vector<vector<float> > &silenceTran,vector<vector<vector<float> > > &digitTran,
          vector<vector<float> > &silenceVari,vector<vector<vector<float> > > &digitVari)
{
    for(int i=1;i<words.size();i++)
    {
        //cout<<"i:"<<i<<endl;
        for(int j=0;j<STATE;j++)   // STATE(7)
        {
            float curMin = FLT_MAX;  // current STATE'S minimum, STATE includes TelState and Silence State
            
            // current Silence state
            for(int n=1;n<3;n++)
            {
                float min = getMin(trellis,i,(j*2),0,n,silenceTran);
                float d = distance(templets[j*2][0][n], words[i], silenceVari[n]);
                trellis[i][j*2][0][n] = min+d;
                
                // update the curMin
                curMin = trellis[i][j*2][0][n]<curMin?trellis[i][j*2][0][n]:curMin;
                // don't do pruning on silence state, because it only has 2 HMM state
                
                // Can go to silence again
                // in order to keep the judgment condition for judging whether digit driving from silence in traceback
                // we keep the trellis[i-1][j*2][0][0] unchanged
                if(n==1 && trellis[i-1][j*2][0][2]<trellis[i-1][j*2][0][0])
                {
                    float temp = trellis[i-1][j*2][0][0];
                    trellis[i-1][j*2][0][0] = trellis[i-1][j*2][0][2];
                    float min = getMin(trellis,i,(j*2),0,n,silenceTran);
                    float d = distance(templets[j*2][0][n], words[i], silenceVari[n]);
                    //cout<<d<<" ";
                    trellis[i][j*2][0][n] = min+d;
                    trellis[i-1][j*2][0][0] = temp;
                }
                //if(n==1)
                //trellis[i][j*2][0][n] += -log(PROB_SILENCE);   // penalty
            }
            
            // use [the final state or dummy state] of silence to update the dummy state of digits in current TEL State
            /* //consider penalty
             float a2 = trellis[i][j*2][0][2];  // already adds penalty
             float a0 = trellis[i][j*2][0][0] - log(1-PROB_SILENCE);
             float temp;
             if(a2 < a0)
             temp = a2;
             else
             temp = a0 - log(1-PROB_SILENCE);
             */
            
            // don't consider penalty:
            float temp = trellis[i][j*2][0][2]<trellis[i][j*2][0][0]?trellis[i][j*2][0][2]:trellis[i][j*2][0][0];
            
            //cout<<"use "<<temp<<" to update digits' dummy state above"<<endl;
            for(int k=0;k<trellis[i][j*2+1].size();k++)
                trellis[i][j*2+1][k][0] = temp;
            
            //else  // go inside next TEL state
            {  
                float curStateMin = FLT_MAX;    // refers to the minimum inside Tel State's las row
                for(int k=0;k<templets[j*2+1].size();k++)    // the k th Digit
                {
                    for(int n=1;n<templets[j*2+1][k].size();n++)  // the n th HMM (including dummy state)
                    {
                        float min,d;
                        if(j==0)
                        {
                            min=getMin(trellis,i,(j*2+1),k,n,digitTran[k+2]);
                            d = distance(templets[j*2+1][k][n], words[i], digitVari[k+2][n]);
                        }
                        else
                        {
                            min=getMin(trellis,i,(j*2+1),k,n,digitTran[k]);
                            d = distance(templets[j*2+1][k][n], words[i], digitVari[k][n]);
                        }
                        trellis[i][j*2+1][k][n] = min+d;
                        
                        //cout<<"curMin: "<<curMin<<"  curValue: "<<min+d<<endl;
                        
                        // update the curMin
                        curMin = trellis[i][j*2+1][k][n]<curMin?trellis[i][j*2+1][k][n]:curMin;
                        
                    }
                    
                }
                
                for(int k=0;k<templets[j*2+1].size();k++)
                {
                    int pos = templets[j*2+1][k].size()-1;
                    curStateMin = trellis[i][j*2+1][k][pos]<curStateMin?trellis[i][j*2+1][k][pos]:curStateMin;
                }
                
                // use curStateMin of current TEL State to update the dummy state of the next SILENCE State
                if(j<STATE-1)
                    trellis[i][(j+1)*2][0][0] = curStateMin;
            }
            
            
        }
    }
    
    float cost;
    vector<int> result = traceback(trellis,silenceTran,digitTran,cost);
    return cost;
}

// real time test
vector<int> dtw(const char *Mean[],const char *Vari[],const char *Tran[],vector<float*> &mfcc)
// for debug
//vector<int> dtw(const char *Mean[],const char *Vari[],const char *Tran[],vector<vector<float> >&mfcc)
{
    vector<vector<float> > silenceVari(3);  //3x39, include dummy
    vector<vector<vector<float> > > digitVari(10);   //10x6x39, include dummy
    makeVari(Vari,silenceVari,digitVari);
    
    vector<vector<float> > silenceTran(3);  // 3(2HMM+dummy) x 3 (direction of transition)
    vector<vector<vector<float> > > digitTran(10);   //10 digit x 6(5HMM+dummy) x 3 direction
    makeTran(Tran,silenceTran,digitTran);
    
    vector<vector<float> > words = makeWord(mfcc);
    
    vector<vector<vector<vector<float> > > >  templets = makeTemplet(Mean);
    vector<vector<vector<vector<float> > > > trellis = makeTrellis(templets,words);
    
    
    float cost = DP (trellis,templets,words,silenceTran,digitTran,silenceVari,digitVari);
    vector<int> result = traceback(trellis,silenceTran,digitTran,cost);
    
    
    return result;
}