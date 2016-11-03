#include "HmmModel.h"

HmmModel::HmmModel(double ***_means, double ***_variances, double **_alpha, double **_transitionCost, int _guassianNum, int _stateNum, int _featureLength) {
	stateNum = _stateNum;
	transitionCost = (double**)malloc(stateNum * sizeof(double*));
	for (int i = 0; i < stateNum; i++) {
		transitionCost[i] = (double*)malloc(stateNum * sizeof(double));
		for (int j = 0; j < stateNum; j++) {
			transitionCost[i][j] = _transitionCost[i][j];
		}
	}
	guassianModels = (GuassianModel**)malloc(stateNum * sizeof(GuassianModel));
	for (int i = 0; i < stateNum; i++) {
		guassianModels[i] = new GuassianModel(_means[i], _variances[i], _alpha[i], _guassianNum, _featureLength);
	}
}