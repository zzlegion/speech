#ifndef _GAUSSIANMODEL_H_
#define _GAUSSIANMODEL_H_

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <vector>

using namespace std;

#define FEATURE_LENGTH (39)
#define M_PI (3.141592653f)
#define EPSILON (500)

class GuassianModel {
private:
	double *logP;
	double *logP_;

	void trainSingleGuassian();
	void trainMixtureGuassian();
	double getNegativeLogLikelihood(vector<double*> points, int pointNum);
	double getLogProbOnSingleGuassian(int index, double *point);
	double getLogSumFromSingle(double *singles);
	void getPosterior(double **posterior, vector<double*> points, int pointNum);
	void updateParameters(double **posterior, vector<double*> points, int pointNum);
	void initializeMixtrueGuassian(double **points, int pointNum);

public:
	double **means;
	double **variances;
	double *alpha;
	int guassianNum;
	int featureLength;

	vector<double*> points;

	GuassianModel(double **_means, double **_variances, double *_alpha, int _guassianNum, int _featureLength);
	GuassianModel(int _guassianNum);
	void trainModel();
	double getCost(double *point);
};

#endif