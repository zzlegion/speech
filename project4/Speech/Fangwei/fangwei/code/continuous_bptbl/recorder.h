#ifndef _RECORDER_H_
#define _RECORDER_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "portaudio.h"
#include "MfccExtractor.h"
#include "readwave.h"
#include "Recognizer.h"
#include <sstream>

using namespace std;

#define SAMPLE_RATE  (8000)
#define FRAMES_PER_BUFFER (400)
#define NUM_SECONDS     (60)
#define NUM_CHANNELS    (1)
#define DITHER_FLAG     (0)

#define OFFSET_THRES (20)
#define SHORTEST_SEGMENT (5)
#define SILENCE_LENGTH (10)  //silence

#define PA_SAMPLE_TYPE  paInt16
typedef short SAMPLE;
#define SAMPLE_SILENCE  (0)

class Recorder {
private:
	static int recordCallback( const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData );
public:
	static void record(bool loopBack, int beamWidth, string modelFile, string silFile);
};

#endif