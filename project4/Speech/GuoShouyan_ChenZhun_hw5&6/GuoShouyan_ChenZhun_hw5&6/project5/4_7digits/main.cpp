#include <stdio.h>
#include "stdlib.h"
#include <math.h>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include "portaudio.h"
#include "readwave.h"
#include "MFCC.h"
#include "dtw_continuous.h"

#define TIME (1)
#define SAMPLE_RATE (44100)
#define FRAMEPRBUFFER (512)

using namespace std;
const char *wavFile1="/Users/guo/Desktop/record1.wav";
const char *wavFile2="/Users/guo/Desktop/record2.wav";
int flag=1, recordTime=0;

typedef struct
{
    vector<short> audio_data;
    int startPoint;
    int begin;
    int time; //how many continuous time not speaking
    int count; //count how many times be callback
}
patest1data;

//calculate the energy of every frame
float Energy(short *inputBuffer,int framesPerBuffer)
{
    float sum=0;
    for(int i=0;i<framesPerBuffer; i++)
    {
        sum=inputBuffer[i]*inputBuffer[i];
    }
    return 10*log(sum+1);
}

//adaptive threshold method
const float adjustment=0.05, forgetfactor=1, threshold=40;
float background=0,level=0;

bool AdaptThre(short *inputBuffer,int framesPerBuffer,int offset)
{
    float current = Energy(inputBuffer,framesPerBuffer);
    bool isSpeech = false;
    level = ((level * forgetfactor) + current) / (forgetfactor+ 1);
    
    if (current < background)
        background = current;
    else
        background += (current - background) * adjustment;
    if (level < background)
        level = background;
    if (level - background > threshold+offset)
        isSpeech = true;
    
    //for test
    /*
     printf( "current=%f, ",current );
     printf( "level=%f, ",level );
     printf( "background=%f \n", background); fflush(stdout);
     */
    return isSpeech;
}

//callback function
float initBG = 0.0f;
static int patest1Callback( const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData )
{
    patest1data *data = (patest1data*)userData;
    short *in = (short*)inputBuffer;
    int framesToCalc = (int)framesPerBuffer;
    int finished = paContinue;
    
    //delete first 10 frames, use 11~20 to set background noise
    if(data->count < 20)
    {
        data->count++;
        if(data->count>9)
            initBG = initBG + Energy(in,framesToCalc);
        if(data->count == 19)
            background=initBG/10;
    }
    //waiting for speaking
    else if(data->begin==0)
    {
        //record
        for(int i=0;i<framesToCalc;i++)
            data->audio_data.push_back(in[i]);
        
        if(AdaptThre(in,framesToCalc, 3))
        {
            printf( "yes.\n" ); fflush(stdout);
            data->time++;
        }
        else
        {
            printf( "no.\n" ); fflush(stdout);
            data->time=0;
        }
        // 44100/512=86, 86*0.1=9
        // speaking for 0.2 seconds, start
        if(data->time > 6)
        {
            data->time=0;
            data->begin=1;
            // preserve the data before
            long size=data->audio_data.size();
            data->startPoint=(size-(6+25)*FRAMEPRBUFFER > 0)?(int)size-(6+25)*FRAMEPRBUFFER:0;
        }
    }
    else
    {
        //detect whether it is speaking
        if(AdaptThre(in,framesToCalc,0))
        {
            printf( "speaking.\n" ); fflush(stdout);
            data->time=0;
        }
        else
        {
            printf( "silent.\n" ); fflush(stdout);
            data->time++;
        }
        // 44100/512=86, 86*1.5=130
        // not speaking for 2 seconds, end
        if(data->time > 130)
        {
            finished=paComplete;
            flag=0;
        }
        recordTime++;  //to calculate the total length
        for(int i=0;i<framesToCalc;i++)
            data->audio_data.push_back(in[i]);
        
    }
    return finished;
}

int main(int argc, char* argv[]);
int main(int argc, char* argv[])
{
    PaStream                *stream;
    PaError                 err;
    patest1data             data;
    PaStreamParameters      inputParameters;
    const PaHostErrorInfo*  herr;
    
    int countForTail=0;
    long tail;
    
    printf("patest1.c\n"); fflush(stdout);
    printf("Ring modulate input for 20 seconds.\n"); fflush(stdout);
    
    /* initialise sinusoidal wavetable */
    data.startPoint=0;
    data.begin = 0;
    data.time=0;
    data.count=0;
    /* initialise portaudio subsytem */
    err = Pa_Initialize();
    
    inputParameters.device = Pa_GetDefaultInputDevice();    /* default input device */
    if (inputParameters.device == paNoDevice) {
        fprintf(stderr,"Error: No input default device.\n");
        goto done;
    }
    inputParameters.channelCount = 1;                       /* mono input */
    inputParameters.sampleFormat = paInt16;           /* 16 bit floating point input */
    inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;
    
    err = Pa_OpenStream(
                        &stream,
                        &inputParameters,
                        NULL,
                        (double)SAMPLE_RATE, /* Samplerate in Hertz. */
                        FRAMEPRBUFFER,       /* Small buffers */
                        paClipOff,           /* We won't output out of range samples so don't bother clipping them. */
                        patest1Callback,
                        &data );
    if( err != paNoError ) goto done;
    
    err = Pa_StartStream( stream );
    if( err != paNoError ) goto done;
    
    printf( "Stream begin.\n" ); fflush(stdout);
    
    //    getc( stdin ); /* wait for input before exiting */
    while(flag);
    
    // cut the tail
    tail=data.audio_data.size();
    cout<<"total length="<<tail<<endl;
    // WriteWave(wavFile1,&data.audio_data[0],(int)tail,SAMPLE_RATE);
    while (countForTail<5) {
        tail=tail-FRAMEPRBUFFER;
        if(AdaptThre(&data.audio_data[tail],FRAMEPRBUFFER,0))
            countForTail++;
        else
            countForTail=0;
        if(tail<FRAMEPRBUFFER) break;
    }
    tail=(tail+FRAMEPRBUFFER*(5+15)<data.audio_data.size())?(tail+FRAMEPRBUFFER*(5+15)):data.audio_data.size();
    cout<<"after prune="<<tail<<endl;
    
    //write to a file
    WriteWave(wavFile2,&data.audio_data[data.startPoint],(int)tail-data.startPoint,SAMPLE_RATE);
    err = Pa_AbortStream( stream );
    if( err != paNoError ) goto done;
    
    printf( "Waiting for stream to complete...\n" );
    /* sleep until playback has finished */
    while( ( err = Pa_IsStreamActive( stream ) ) == 1 ) Pa_Sleep(1000);
    if( err < 0 ) goto done;
    
    err = Pa_CloseStream( stream );
    if( err != paNoError ) goto done;
    
done:
    Pa_Terminate();
    
    if( err != paNoError )
    {
        fprintf( stderr, "An error occured while using portaudio\n" );
        if( err == paUnanticipatedHostError )
        {
            fprintf( stderr, " unanticipated host error.\n");
            herr = Pa_GetLastHostErrorInfo();
            if (herr)
            {
                fprintf( stderr, " Error number: %ld\n", herr->errorCode );
                if (herr->errorText)
                    fprintf( stderr, " Error text: %s\n", herr->errorText );
            }
            else
                fprintf( stderr, " Pa_GetLastHostErrorInfo() failed!\n" );
        }
        else
        {
            fprintf( stderr, " Error number: %d\n", err );
            fprintf( stderr, " Error text: %s\n", Pa_GetErrorText( err ) );
        }
        
        err = 1;          /* Always return 0 or 1, but no other return codes. */
    }
    
    printf( "record finished\n" );
    //.....................MFCC..........................
    const char *Mean[]={
        "/Users/guo/Desktop/project6/resultA/mean0.txt",
        "/Users/guo/Desktop/project6/resultA/mean1.txt",
        "/Users/guo/Desktop/project6/resultA/mean2.txt",
        "/Users/guo/Desktop/project6/resultA/mean3.txt",
        "/Users/guo/Desktop/project6/resultA/mean4.txt",
        "/Users/guo/Desktop/project6/resultA/mean5.txt",
        "/Users/guo/Desktop/project6/resultA/mean6.txt",
        "/Users/guo/Desktop/project6/resultA/mean7.txt",
        "/Users/guo/Desktop/project6/resultA/mean8.txt",
        "/Users/guo/Desktop/project6/resultA/mean9.txt",
        "/Users/guo/Desktop/project6/resultA/meansilence.txt"
    };
    const char *Vari[]={
        "/Users/guo/Desktop/project6/resultA/vari0.txt",
        "/Users/guo/Desktop/project6/resultA/vari1.txt",
        "/Users/guo/Desktop/project6/resultA/vari2.txt",
        "/Users/guo/Desktop/project6/resultA/vari3.txt",
        "/Users/guo/Desktop/project6/resultA/vari4.txt",
        "/Users/guo/Desktop/project6/resultA/vari5.txt",
        "/Users/guo/Desktop/project6/resultA/vari6.txt",
        "/Users/guo/Desktop/project6/resultA/vari7.txt",
        "/Users/guo/Desktop/project6/resultA/vari8.txt",
        "/Users/guo/Desktop/project6/resultA/vari9.txt",
        "/Users/guo/Desktop/project6/resultA/varisilence.txt"
    };
    const char *Tran[]={
        "/Users/guo/Desktop/project6/resultA/tran0.txt",
        "/Users/guo/Desktop/project6/resultA/tran1.txt",
        "/Users/guo/Desktop/project6/resultA/tran2.txt",
        "/Users/guo/Desktop/project6/resultA/tran3.txt",
        "/Users/guo/Desktop/project6/resultA/tran4.txt",
        "/Users/guo/Desktop/project6/resultA/tran5.txt",
        "/Users/guo/Desktop/project6/resultA/tran6.txt",
        "/Users/guo/Desktop/project6/resultA/tran7.txt",
        "/Users/guo/Desktop/project6/resultA/tran8.txt",
        "/Users/guo/Desktop/project6/resultA/tran9.txt",
        "/Users/guo/Desktop/project6/resultA/transilence.txt"
    };
    
    vector<float*> mfcc=MFCC(&data.audio_data[0],data.startPoint, (int)tail);
    
    //print out the result
    //.........................................
    ofstream outfile("/Users/guo/Desktop/output.txt");
    for(int i=0;i<mfcc.size();i++)
    {
        for(int j=0;j<39;j++)
        {
            outfile<<mfcc[i][j]<<" ";
        }
        outfile<<endl;
    }
    outfile.close();
    //.....................................
    
    //int result= dtw(fileName,mfcc);
    vector<int> result =dtw(Mean,Vari,Tran,mfcc);
    cout << "Finish!\n";
    for(int i=result.size()-1;i>=0;i--)
    {
        if(result[i]!=-1)
            cout<<result[i]<<" ";
    }
    return 0;
    
    
}