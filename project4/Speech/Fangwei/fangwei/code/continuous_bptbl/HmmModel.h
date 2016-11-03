#include "GuassianModel.h"

class HmmModel {
public:
	HmmModel(double ***_means, double ***_variances, double **_alpha, double **_transitionCost, int _guassianNum, int _stateNum, int _featureLength);

	int stateNum;
	double **transitionCost;
	GuassianModel **guassianModels;
};