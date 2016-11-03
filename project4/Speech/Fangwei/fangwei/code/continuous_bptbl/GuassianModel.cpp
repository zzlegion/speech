#include "GuassianModel.h"

GuassianModel::GuassianModel(double **_means, double **_variances, double *_alpha, int _guassianNum, int _featureLength) {
	guassianNum = _guassianNum;
	featureLength = _featureLength;
	means = (double**)malloc(guassianNum * sizeof(double*));
	variances = (double**)malloc(guassianNum * sizeof(double*));
	for (int i = 0; i < guassianNum; i++) {
		means[i] = (double*)malloc(featureLength * sizeof(double));
		variances[i] = (double*)malloc(featureLength * sizeof(double));
		for (int j = 0; j < featureLength; j++) {
			means[i][j] = _means[i][j];
			variances[i][j] = _variances[i][j];
		}
	}
	alpha = (double*)malloc(guassianNum * sizeof(double));
	for (int i = 0; i < guassianNum; i++) {
		alpha[i] = _alpha[i];
	}

	logP = (double*)malloc(guassianNum * sizeof(double));
	logP_ = (double*)malloc(guassianNum * sizeof(double));
}

GuassianModel::GuassianModel(int _guassianNum) {
	guassianNum = _guassianNum;
	means = (double**)malloc(guassianNum * sizeof(double*));
	variances = (double**)malloc(guassianNum * sizeof(double*));
	for (int i = 0; i < guassianNum; i++) {
		means[i] = (double*)malloc(FEATURE_LENGTH * sizeof(double));
		variances[i] = (double*)malloc(FEATURE_LENGTH * sizeof(double));
	}
	alpha = (double*)malloc(guassianNum * sizeof(double));
	for (int i = 0; i < guassianNum; i++) {
		alpha[i] = 1.0f / guassianNum;
	}
}

void GuassianModel::trainModel(double **points, int pointNum) {
	if (guassianNum == 1)
		trainSingleGuassian(points, pointNum);
	else
		trainMixtureGuassian(points, pointNum);
}

double GuassianModel::getCost(double *point) {
	/*double *logP = (double*)malloc(guassianNum * sizeof(double));*/
	for (int i = 0; i < guassianNum; i++) {
		logP[i] = getLogProbOnSingleGuassian(i, point);
	}

	double result = getLogSumFromSingle(logP);
	/*if (result >= 0)
		result = 0;*/
	//free(logP);
	return -result;
}

double GuassianModel::getLogProbOnSingleGuassian(int index, double *point) {
	double result = 0;
	for (int j = 0; j < FEATURE_LENGTH; j++) {
		/*if (result < 0)
			printf("_negative!");*/
		result += ((point[j] - means[index][j]) * (point[j] - means[index][j]) / variances[index][j]);
		/*if (result < 0)
			printf("negative!");*/
		if (2 * M_PI * variances[index][j] <= 0)
			printf("error\n");
		result += log(2 * M_PI * variances[index][j]);
		/*if (result < 0)
			printf("_negative_!");*/
	}
	result *= -0.5f;
	/*if (result >= 0)
		result = 0;*/
	result += log(alpha[index]);
	if (alpha[index] <= 0)
			printf("error\n");
	return result;
}

double GuassianModel::getLogSumFromSingle(double *_logP) {
	//double *logP = (double*)malloc(guassianNum * sizeof(double));
	for (int i = 0; i < guassianNum; i++)
		logP_[i] = _logP[i];
	//sort
	for (int i = 0; i < guassianNum - 1; i++) {
		double max = logP_[0];
		int index = 0;
		for (int j = 1; j < guassianNum - i; j++) {
			if (max < logP_[j]) {
				max = logP_[j];
				index = j;
			}
		}
		double temp = logP_[guassianNum - i - 1];
		logP_[guassianNum - i - 1] = max;
		logP_[index] = temp;
	}
	double result = logP_[0];
	for (int i = 1; i < guassianNum; i++) {
		double temp = result - logP_[i];
		temp = exp(temp);
		temp = log(1.0f + temp);
		result = temp + logP_[i];
	}
	return (double)result;
}

void GuassianModel::trainSingleGuassian(double **points, int pointNum) {
	for (int i = 0; i < guassianNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			means[i][j] = 0;
			variances[i][j] = 0;
		}
	}

	for (int i = 0; i < pointNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			means[0][j] += points[i][j];
		}
	}
	for (int i = 0; i < FEATURE_LENGTH; i++) {
		means[0][i] /= pointNum;
	}
	for (int i = 0; i < pointNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			variances[0][j] += (points[i][j] - means[0][j]) * (points[i][j] - means[0][j]);
		}
	}
	for (int i = 0; i < FEATURE_LENGTH; i++) {
		variances[0][i] /= pointNum;
	}
}

double gaussrand()
{
    double V1, V2, S;
    int phase = 0;
    double X;
     
    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
         
    phase = 1 - phase;
 
    return X;
}

void GuassianModel::initializeMixtrueGuassian(double **points, int pointNum) {
	int *label = (int*)malloc(pointNum * sizeof(int));
	int *componentNum = (int*)malloc(guassianNum * sizeof(int));

	double previousCost = 0;
	double currentCost = 0;

	do {
		previousCost = currentCost;
		// assign
		currentCost = 0;
		for (int i = 0; i < guassianNum; i++) {
			componentNum[i] = 0;
		}
		for (int i = 0; i < pointNum; i++) {
			double min = -getLogProbOnSingleGuassian(0, points[i]);
			label[i] = 0;
			for (int j = 1; j < guassianNum; j++) {
				double temp  = -getLogProbOnSingleGuassian(j, points[i]);
				if (temp < min) {
					min = temp;
					label[i] = j;
				}
			}
			componentNum[label[i]]++;
			currentCost += min;
		}

		for (int i = 0; i < guassianNum; i++) {
			if (componentNum[i] > 1)
				continue;
			int index = 0;
			for (int j = 0; j < pointNum; j++) {
				if (componentNum[label[j]] > 2)
					index = j;
			}
			double min1 = -getLogProbOnSingleGuassian(i, points[index]);
			double min2 = min1;
			int index1 = index;
			int index2 = index;

			for (int j = index; j < pointNum; j++) {
				double temp = -getLogProbOnSingleGuassian(i, points[j]);
				if (temp < min2 && componentNum[label[j]] > 2) {
					min2 = temp;
					index2 = j;
					if (min2 < min1) {
						min2 = min1;
						index2 = index1;
						min1 = temp;
						index1 = j;
					}
				}
				
			}

			componentNum[label[index1]]--;
			componentNum[label[index2]]--;
			componentNum[i] += 2;
			label[index1] = i;
			label[index2] = i;
		}

		for (int i = 0; i < guassianNum; i++) {
			if (componentNum[i] == 0 || componentNum[i] == 1)
				printf("component zero!\n");
		}

		//update
		for (int i = 0; i < guassianNum; i++) {
			for (int j = 0; j < FEATURE_LENGTH; j++) {
				means[i][j] = 0;
				variances[i][j] = 0;
			}
		}
		for (int i = 0; i < pointNum; i++) {
			for (int j = 0; j < FEATURE_LENGTH; j++) {
				means[label[i]][j] += points[i][j];
			}
		}
		for (int i = 0; i < guassianNum; i++) {
			for (int j = 0; j < FEATURE_LENGTH; j++) {
				means[i][j] /= componentNum[i];
			}
		}
		for (int i = 0; i < pointNum; i++) {
			for (int j = 0; j < FEATURE_LENGTH; j++) {
				variances[label[i]][j] += (points[i][j] - means[label[i]][j]) * (points[i][j] - means[label[i]][j]);
			}
		}
		for (int i = 0; i < guassianNum; i++) {
			for (int j = 0; j < FEATURE_LENGTH; j++) {
				variances[i][j] /= componentNum[i];
			}
		}
		//printf("init_cost: %f\n", currentCost);
	} while (abs(currentCost - previousCost) >= EPSILON);
	//printf("cost: %f\n", currentCost);
}

void GuassianModel::trainMixtureGuassian(double **points, int pointNum) {
	for (int i = 0; i < guassianNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			means[i][j] = gaussrand();//(double)(rand() + 1) / RAND_MAX;
			variances[i][j] = abs(gaussrand()) + 0.1f;//(double)(rand() + 1) / RAND_MAX;
		}
	}
	//initializeMixtrueGuassian(points, pointNum);
	
	double previousCost = 0;
	double currentCost = 0;

	double **posterior = (double**)malloc(pointNum * sizeof(double*));
	for (int i = 0; i < pointNum; i++) {
		posterior[i] = (double*)malloc(guassianNum * sizeof(double));
	}

	do {
		previousCost = currentCost;
		// E step
		getPosterior(posterior, points, pointNum);
		// M step
		updateParameters(posterior, points, pointNum);
		currentCost = getNegativeLogLikelihood(points, pointNum);
		//printf("cost: %f\n", currentCost);
	} while(abs(currentCost - previousCost >= EPSILON));
}

void GuassianModel::getPosterior(double **posterior, double **points, int pointNum) {
	for (int i = 0; i < pointNum; i++) {
		for (int j = 0; j < guassianNum; j++) {
			posterior[i][j] = getLogProbOnSingleGuassian(j, points[i]);
		}
		double logSum = getLogSumFromSingle(posterior[i]);
		for (int j = 0; j < guassianNum; j++) {
			posterior[i][j] -= logSum;
			posterior[i][j] = exp(posterior[i][j]);
			if (abs(posterior[i][j]) <= 1e-30)
				posterior[i][j] = 0.001f;
		}
	}
}

void GuassianModel::updateParameters(double **posterior, double **points, int pointNum) {
	double *sumOfPosterior = (double*)malloc(guassianNum * sizeof(double));
	for (int i = 0; i < guassianNum; i++) {
		sumOfPosterior[i] = 0;
		for (int j = 0; j < pointNum; j++) {
			sumOfPosterior[i] += posterior[j][i];
		}
	}
	// update alpha
	for (int i = 0; i < guassianNum; i++) {
		alpha[i] = sumOfPosterior[i] / pointNum;
	}
	// update mean
	for (int i = 0; i < guassianNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			double sum = 0;
			for (int k = 0; k < pointNum; k++) {
				sum += (points[k][j] * posterior[k][i]);
			}
			means[i][j] = sum / sumOfPosterior[i];
		}
	}
	// update variance
	for (int i = 0; i < guassianNum; i++) {
		for (int j = 0; j < FEATURE_LENGTH; j++) {
			double sum = 0;
			for (int k = 0; k < pointNum; k++) {
				sum += ((points[k][j] - means[i][j]) * (points[k][j] - means[i][j]) * posterior[k][i]);
			}
			variances[i][j] = sum / sumOfPosterior[i];
		}
	}
}

double GuassianModel::getNegativeLogLikelihood(double **points, int pointNum) {
	double result = 0;
	for (int i = 0; i < pointNum; i++) {
		result += getCost(points[i]);
	}
	return result;
}