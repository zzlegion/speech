#include <stdio.h>
#include "MfccExtractor.h"

double** MfccExtractor::getMfccFromWav(const char* filename, int *frameNum, double ***mfcc) {
	int sampleNum, sampleRate;
	SAMPLE *samples = ReadWave(filename, &sampleNum, &sampleRate);
	return getMfcc(samples, sampleNum, sampleRate, frameNum, mfcc);
}

double** MfccExtractor::getMfcc(SAMPLE *samples, int sampleNum, int sampleRate, int *frameNum, double ***mfcc) {
	double *preemSamples = preemphasis(samples, sampleNum);
	//int frameNum = 0;
	double **fzwSamples = frameAndZeroPaddingThenWindow(preemSamples, sampleNum, frameNum);
	double **dftSamples = dft(fzwSamples, *frameNum);
	double **melSpectrum = melWarpingWithFilterAndLog(dftSamples, *frameNum, sampleRate);
	double **cepstra = dct(melSpectrum, *frameNum);
	/*for (int i = 0; i < DCT_PRINCIPAL_NUM; i++)
		printf();*/
	addTemporalFeatures(cepstra, *frameNum, mfcc);
	standardNormalize(*mfcc, *frameNum, 3 * DCT_PRINCIPAL_NUM);
	//standardNormalize(*mfcc, *frameNum, DCT_PRINCIPAL_NUM);
	return *mfcc;
}

double* MfccExtractor::preemphasis(SAMPLE* samples, int sampleNum) {
	double *result = (double*)malloc(sampleNum * sizeof(double));
	result[0] = (double)samples[0];
	for (int i = 1; i < sampleNum; i++) {
		result[i] = (double)samples[i] -  PREEM_ALPHA * samples[i - 1];
	}
	return result;
}

double** MfccExtractor::frameAndZeroPaddingThenWindow(double *preemSamples, int sampleNum, int *frameNum) {
	*frameNum = (int)ceil((double)(sampleNum - FRAME_WIDTH) / FRAME_SHIFT) + 1;
	double **result = (double**)malloc(*frameNum * sizeof(double*));
	for (int i = 0; i < *frameNum; i++) {
		result[i] = (double*)calloc(FRAME_PADDING_WIDTH, sizeof(double));
		int start = i * FRAME_SHIFT;
		int length = ((sampleNum - start) < FRAME_WIDTH ? (sampleNum - start) : FRAME_WIDTH);
		memcpy(&result[i][(FRAME_PADDING_WIDTH - FRAME_WIDTH) / 2], &preemSamples[start], length * sizeof(double));
	}
	for (int i = 0; i < *frameNum; i++) {
		int start = (FRAME_PADDING_WIDTH - FRAME_WIDTH) / 2;
		
		for (int j = 0; j < FRAME_WIDTH; j++) {
			double hamming = 0.54f - 0.46f * cos(2 * PI * j / FRAME_WIDTH);
			result[i][start + j] = result[i][start + j] * hamming;
		}
		
	}
	return result;
}

double** MfccExtractor::dft(double **fzwSamples, int frameNum) {
	int outNum = FRAME_PADDING_WIDTH / 2 + 1;
	fftw_complex *out = (fftw_complex*)fftw_malloc(outNum * sizeof(fftw_complex));
	double *in = (double*)malloc(FRAME_PADDING_WIDTH * sizeof(double));
	fftw_plan plan_forward = fftw_plan_dft_r2c_1d(FRAME_PADDING_WIDTH, in, out, FFTW_ESTIMATE);

	double **result = (double**)malloc(frameNum * sizeof(double*));
	for (int i = 0; i < frameNum; i++) {
		result[i] = (double*)malloc(outNum * sizeof(double));
		for (int j = 0; j < FRAME_PADDING_WIDTH; j++) {
			in[j] = (double)fzwSamples[i][j];
		}
		fftw_execute(plan_forward);
		for (int j = 0; j < outNum; j++) {
			result[i][j] = out[j][0] * out[j][0] + out[j][1] * out[j][1];
		}
	}
	return result;
}

double** MfccExtractor::melWarpingWithFilterAndLog(double **dftSamples, int frameNum, int sampleRate) {
	int outNum = FRAME_PADDING_WIDTH / 2 + 1;
	double nyqMax = sampleRate / 2;
	double melMax = melTransform(nyqMax);
	double delta = melMax / (MEL_FILTER_NUM + 1);

	double **result = (double**)malloc(frameNum * sizeof(double*));
	for (int i = 0; i < frameNum; i++) {
		result[i] = (double*)calloc(MEL_FILTER_NUM, sizeof(double));
		for (int j = 0; j < outNum; j++) {
			double mefFreq = melTransform(nyqMax * j / (outNum - 1));
			int slot = (int)floor(mefFreq / delta);
			double remainder = mefFreq - delta * slot;
			if (slot - 1 >= 0) {
				double triangleScalar = (delta - remainder) / delta;
				result[i][slot - 1] += triangleScalar * dftSamples[i][j];
			}
			if (slot < MEL_FILTER_NUM) {
				double triangleScalar = remainder / delta;
				result[i][slot - 1] += triangleScalar * dftSamples[i][j];
			}
		}
		for (int j = 0; j < MEL_FILTER_NUM; j++) {
			result[i][j] = (double)log(result[i][j]);
		}
	}
	return result;
}

double** MfccExtractor::dct(double **melSpecturm, int frameNum) {
	double *out = (double*)malloc(MEL_FILTER_NUM * sizeof(double));
	double *in = (double*)malloc(MEL_FILTER_NUM * sizeof(double));
	fftw_plan dct = fftw_plan_r2r_1d(MEL_FILTER_NUM, in, out, FFTW_REDFT10, FFTW_MEASURE);

	double **result = (double**)malloc(frameNum * sizeof(double*));
	for (int i = 0; i < frameNum; i++) {
		result[i] = (double*)malloc(DCT_PRINCIPAL_NUM * sizeof(double));
		for (int j = 0; j < MEL_FILTER_NUM; j++) {
			in[j] = melSpecturm[i][j];
		}
		fftw_execute(dct);
		for (int j = 0; j < DCT_PRINCIPAL_NUM; j++) {
			result[i][j] = out[j];
		}
	}
	return result;
}

double** MfccExtractor::addTemporalFeatures(double **mfcc, int frameNum, double ***result) {
	*result = (double**)malloc(frameNum * sizeof(double*));
	for (int i = 0; i < frameNum; i++) {
		(*result)[i] = (double*)calloc(3 * DCT_PRINCIPAL_NUM, sizeof(double));
		//(*result)[i] = (double*)calloc(DCT_PRINCIPAL_NUM, sizeof(double));
		memcpy((*result)[i], mfcc[i], DCT_PRINCIPAL_NUM * sizeof(double));
		if (i - 1 >= 0 && i + 1 < frameNum) {
			for (int j = 0; j < DCT_PRINCIPAL_NUM; j++) {
				(*result)[i][DCT_PRINCIPAL_NUM + j] = mfcc[i + 1][j] - mfcc[i - 1][j];
			}
		}
	}
	for (int i = 2; i < frameNum - 2; i++) {
		for (int j = 0; j < DCT_PRINCIPAL_NUM; j++) {
			(*result)[i][2 * DCT_PRINCIPAL_NUM + j] = (*result)[i + 1][DCT_PRINCIPAL_NUM + j] - (*result)[i - 1][DCT_PRINCIPAL_NUM + j];
		}
	}
	return *result;
}

void MfccExtractor::standardNormalize(double **mfcc, int frameNum, int featureLength) {
	double *mean = (double*)calloc(featureLength, sizeof(double));
	double *sd = (double*)calloc(featureLength, sizeof(double));
	for (int i = 0; i < frameNum; i++) {
		for (int j = 0; j < featureLength; j++)
			mean[j] += mfcc[i][j];
	}
	for (int i = 0; i < featureLength; i++) {
		mean[i] /= frameNum;
	}
	for (int i = 0; i < frameNum; i++) {
		for (int j = 0; j < featureLength; j++) {
			sd[j] += (mfcc[i][j] - mean[j]) * (mfcc[i][j] - mean[j]);
		}
	}
	for (int i = 0; i < featureLength; i++) {
		sd[i] /= frameNum;
		sd[i] = sqrt(sd[i]);
	}
	for (int i = 0; i < frameNum; i++) {
		for (int j = 0; j < featureLength; j++) {
			mfcc[i][j] = (mfcc[i][j] - mean[j]) / sd[j];
		}
	}
}

double MfccExtractor::melTransform(double value) {
	return 2595 * log10(1 + value / 700);
}