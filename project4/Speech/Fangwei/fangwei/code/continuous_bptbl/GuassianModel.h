#ifndef _GAUSSIANMODEL_H_
#define _GAUSSIANMODEL_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define FEATURE_LENGTH (39)
#define M_PI (3.141592653f)
#define EPSILON (100)

class GuassianModel {
private:
	int guassianNum;
	int featureLength;
	double *logP;
	double *logP_;

	void trainSingleGuassian(double **points, int pointNum);
	void trainMixtureGuassian(double **points, int pointNum);
	double getNegativeLogLikelihood(double **points, int pointNum);
	double getLogProbOnSingleGuassian(int index, double *point);
	double getLogSumFromSingle(double *singles);
	void getPosterior(double **posterior, double **points, int pointNum);
	void updateParameters(double **posterior, double **points, int pointNum);
	void initializeMixtrueGuassian(double **points, int pointNum);

public:
	double **means;
	double **variances;
	double *alpha;
	GuassianModel(double **_means, double **_variances, double *_alpha, int _guassianNum, int _featureLength);
	GuassianModel(int _guassianNum);
	void trainModel(double **points, int pointNum);
	double getCost(double *point);
};

#endif