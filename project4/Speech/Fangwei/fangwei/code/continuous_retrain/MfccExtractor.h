#ifndef _MFCCEXTRACTOR_H_
#define _MFCCEXTRACTOR_H_

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "readwave.h"
#include "fftw3.h"

#define FRAME_WIDTH (400)
#define FRAME_PADDING_WIDTH (512)
#define FRAME_SHIFT (160)
#define PREEM_ALPHA (0.95f)
#define MEL_FILTER_NUM (40)
#define DCT_PRINCIPAL_NUM (13)
#define PI (3.141592653f)

typedef short SAMPLE;

class MfccExtractor {
private:
	static double* preemphasis(SAMPLE* samples, int sampleNum);
	static double** frameAndZeroPaddingThenWindow(double *preemSamples, int sampleNum, int *frameNum);
	static double** dft(double **fzwSamples, int frameNum);
	static double** melWarpingWithFilterAndLog(double **dftSamples, int frameNum, int sampleRate);
	static double** dct(double **melSpecturm, int frameNum);
	static double** addTemporalFeatures(double **mfcc, int frameNum, double ***result);
	static void standardNormalize(double **mfcc, int frameNum, int featureLength);
	static double melTransform(double value);

public:
	static double** getMfcc(SAMPLE* samples, int sampleNum, int sampleRate, int *frameNum, double ***mfcc);
	static double** getMfccFromWav(const char* filename, int *frameNum, double ***mfcc);
};

#endif