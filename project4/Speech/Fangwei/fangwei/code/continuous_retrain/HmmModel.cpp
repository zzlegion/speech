#include "HmmModel.h"

HmmModel::HmmModel(double ***_means, double ***_variances, double **_alpha, double **_transitionCost, int _guassianNum, int _stateNum, int _featureLength) {
	stateNum = _stateNum;
	occurNum = 0;
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

void HmmModel::clearPoints() {
	for (int i = 0; i < stateNum; i++) {
		//for (int j = 0; j < guassianModels[i]->guassianNum; j++) {
		//	/*if (guassianModels[i][j].points.size() > 0)
		//		guassianModels[i][j].points.clear();*/
		//	for (int k = 0; k < guassianModels[i][j].points.size(); k++)
		//		guassianModels[i][j].points.pop_back();
		//}
		guassianModels[i]->points.clear();
	}
}

void HmmModel::trainTransitionCost() {
	for (int i = 0; i < stateNum; i++) {
		int pointNum = guassianModels[i]->points.size();
		transitionCost[i][i] = -log(((double)pointNum - occurNum) / pointNum);
		if (i + 1 >= stateNum)
			continue;
		transitionCost[i][i + 1] = -log((double)occurNum / pointNum);
	}
}