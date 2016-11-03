#ifndef _HMMMODEL_H_
#define _HMMMODEL_H_

#include "GuassianModel.h"

class HmmModel {
public:
	HmmModel(double ***_means, double ***_variances, double **_alpha, double **_transitionCost, int _guassianNum, int _stateNum, int _featureLength);
	void trainTransitionCost();
	void clearPoints();

	int occurNum;
	int stateNum;
	double **transitionCost;
	GuassianModel **guassianModels;
};

#endif