#include "SegKmeans.h"

SegKmeans::SegKmeans() {}

//SegKmeans::SegKmeans(double ***_templates, int *_tFrameNum, int _templateNum) {
//	templateNum = _templateNum;
//
//	// frame number
//	tFrameNum = (int*)malloc(templateNum * sizeof(int));
//	memcpy(tFrameNum, _tFrameNum, templateNum * sizeof(int));
//
//	// templates
//	templates = (double***)malloc(templateNum * sizeof(double**)); 
//	for (int i = 0; i < templateNum; i++) {
//		templates[i] = (double**)malloc(tFrameNum[i] * sizeof(double*));
//		for (int j = 0; j < tFrameNum[i]; j++) {
//			templates[i][j] = (double*)malloc(FEATURE_LENGTH * sizeof(double));
//			memcpy(templates[i][j], _templates[i][j], FEATURE_LENGTH * sizeof(double));
//		}
//	}
//
//	// boundary
//	boundaries = (int**)malloc(templateNum * sizeof(int*));
//	for (int i = 0; i < templateNum; i++) {
//		boundaries[i] = (int*)malloc(STATE_NUM * sizeof(int));
//		int segLength = (tFrameNum[i] % STATE_NUM == 0) ? (int)(tFrameNum[i] / STATE_NUM) : (int)(floor((double)tFrameNum[i] / STATE_NUM) + 1);
//		for (int j = 0; j < STATE_NUM - 1; j++) {
//			boundaries[i][j] = (j + 1) * segLength - 1;
//		}
//		boundaries[i][STATE_NUM - 1] = tFrameNum[i] - 1;
//	}
//}

void SegKmeans::segmentalKmeans(map<string, HmmModel*> &hmms, 
	map<string, vector<double**>*> &trainingData, 
	map<string, vector<int>*> &frameLengths) {
		
	//init boundary
	map<string, vector<int*>> boundaries;

	/*for (int i = 0; i < hmms["0"]->stateNum; i++) {
		printf("%d\n", hmms["0"]->guassianModels[i]->featureLength);
	}*/

	for (map<string, vector<double**>*>::iterator iter = trainingData.begin(); iter != trainingData.end(); iter++) {
		vector<int*> boundary;
		for (int i = 0; i < iter->second->size(); i++) {
			int *bound = (int*)malloc((iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2) * sizeof(int));
			boundary.push_back(bound);
		}
		boundaries[iter->first] = boundary;
	}

	//models
	map<string, GuassianModel**> models;
	for (map<string, vector<double**>*>::iterator iter = trainingData.begin(); iter != trainingData.end(); iter++) {
		GuassianModel **model = (GuassianModel**)malloc((iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2) * sizeof(GuassianModel*));
		vector<string> sequence;
		sequence.push_back("sil");
		for (int i = 0; i < iter->first.length(); i++) {
			sequence.push_back(string(1, iter->first[i]));
		}
		sequence.push_back("sil");
		int scan = 0;
		for (int i = 0; i < sequence.size(); i++) {
			hmms[sequence[i]]->occurNum++;
			for (int j = 0; j < hmms[sequence[i]]->stateNum; j++) {
				model[scan++] = hmms[sequence[i]]->guassianModels[j];
			}
		}
		models[iter->first] = model;
	}

	//transitionCost
	map<string, double**> transitionCosts;
	for (map<string, vector<double**>*>::iterator iter = trainingData.begin(); iter != trainingData.end(); iter++) {
		int totalLength = (iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2);
		double **transitionCost = (double**)malloc(totalLength * sizeof(double*));
		vector<string> sequence;
		sequence.push_back("sil");
		for (int i = 0; i < iter->first.length(); i++) {
			sequence.push_back(string(1, iter->first[i]));
		}
		sequence.push_back("sil");
		int scan = 0;
		for (int i = 0; i < sequence.size(); i++) {
			for (int j = 0; j < hmms[sequence[i]]->stateNum; j++) {
				transitionCost[scan] = (double*)malloc(totalLength * sizeof(double));
				scan++;
			}
		}
		transitionCosts[iter->first] = transitionCost;
	}

	for (int i = 0; i < hmms["0"]->stateNum; i++) {
		printf("%d\n", hmms["0"]->guassianModels[i]->featureLength);
	}

	double previousCost = 0;
	double currentCost = 0;

	int okNum = 0;

	printf("kmeans...\n");
	int cnt = 0;
	do{
		cnt++;
		previousCost = currentCost;

		fillTransitionCostTable(hmms, transitionCosts);
		for (map<string, HmmModel*>::iterator iter = hmms.begin(); iter != hmms.end(); iter++) {
			iter->second->clearPoints();
		}

		currentCost = updateBoundaryAndGetCost(models, trainingData, frameLengths, boundaries, transitionCosts);
		printf("round %d, cost %f\n", cnt, currentCost);
		if (abs(currentCost - previousCost) < EPSILON)
			okNum++;
		else
			okNum = 0;
		if (okNum > 1)
			break;

		updateModelAndTransitionCost(models, trainingData, hmms, boundaries, transitionCosts);
	} while(true);
}

double SegKmeans::updateBoundaryAndGetCost(map<string, GuassianModel**> &models, 
		map<string, vector<double**>*> &trainingData, 
		map<string, vector<int>*> &frameLengths,
		map<string, vector<int*>> &boundaries,
		map<string, double**> &transitionCosts) {
	
	double cost = 0;
	for (map<string, vector<double**>*>::iterator iter = trainingData.begin(); iter != trainingData.end(); iter++) {
		int totalLength = (iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2);
		for (int i = 0; i < iter->second->size(); i++) {
			levenshtein *lev = new levenshtein();
			cost += lev->fillTrellisForKmeansAndReturnCost(models[iter->first], transitionCosts[iter->first], totalLength, (*(iter->second))[i], (*(frameLengths[iter->first]))[i]);
			int pathLength = 0;
			int **path = lev->getAlignment(0, totalLength, (*(frameLengths[iter->first]))[i], &pathLength);
			for (int j = pathLength - 1; j >= 0; j--) {
				(boundaries[iter->first])[i][path[j][0]] = path[j][1];
			}
		}
	}
	return cost;
}

void SegKmeans::updateModelAndTransitionCost(map<string, GuassianModel**> &models, 
		map<string, vector<double**>*> &trainingData, 
		map<string, HmmModel*> &hmms,
		map<string, vector<int*>> &boundaries,
		map<string, double**> &transitionCosts) {

	for (map<string, vector<double**>*>::iterator iter = trainingData.begin(); iter != trainingData.end(); iter++) {
		int totalLength = (iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2);
		for (int i = 0; i < iter->second->size(); i++) {
			for (int j = 0; j < totalLength; j++) {
				int start = (j - 1 < 0 ? 0 : (boundaries[iter->first][i][j - 1] + 1));
				int end = boundaries[iter->first][i][j];
				for (int m = start; m <= end; m++) {
					models[iter->first][j]->points.push_back((*(trainingData[iter->first]))[i][m]);
				}
			}
		}
	}

	for (map<string, HmmModel*>::iterator iter = hmms.begin(); iter != hmms.end(); iter++) {
		for (int i = 0; i < iter->second->stateNum; i++) {
			iter->second->guassianModels[i]->trainModel();
		}
		iter->second->trainTransitionCost();
	}
}

void SegKmeans::fillTransitionCostTable(map<string, HmmModel*> &hmms, map<string, double**> &transitionCosts) {
	for (map<string, double**>::iterator iter = transitionCosts.begin(); iter != transitionCosts.end(); iter++) {
		int totalLength = (iter->first.length() * STATE_NUM + SIL_STATE_NUM * 2);
		
		vector<string> sequence;
		sequence.push_back("sil");
		for (int i = 0; i < iter->first.length(); i++) {
			sequence.push_back(string(1, iter->first[i]));
		}
		sequence.push_back("sil");

		int scan = 0;
		
		for (int i = 0; i < sequence.size(); i++) {
			for (int j = 0; j < hmms[sequence[i]]->stateNum; j++) {
				transitionCosts[iter->first][scan][scan] = hmms[sequence[i]]->transitionCost[j][j];
				if (scan < totalLength - 1) {
					if (j < hmms[sequence[i]]->stateNum - 1) {
						transitionCosts[iter->first][scan][scan + 1] = hmms[sequence[i]]->transitionCost[j][j + 1];
					}
					else {
						transitionCosts[iter->first][scan][scan + 1] = PENALTY;
					}
				}
				scan++;
			}
		}
	}
}