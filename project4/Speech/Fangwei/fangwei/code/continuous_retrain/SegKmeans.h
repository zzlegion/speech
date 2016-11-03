#ifndef _SEGKMEANS_H_
#define _SEGKMEANS_H_

#include <math.h>
#include "levenshtein.h"
//#include "GuassianModel.h"
#include "HmmModel.h"
#include <vector>
#include <map>
#include <string>

using namespace std;

#define STATE_NUM (5)
#define POINT_NUM (1000)
#define SIL_STATE_NUM (2)
#define PENALTY (200)

class SegKmeans {
private:
	/*double ***templates;
	int *tFrameNum;
	int templateNum;
	int **boundaries;*/

	static void updateModelAndTransitionCost(map<string, GuassianModel**> &models, 
		map<string, vector<double**>*> &trainingData, 
		map<string, HmmModel*> &hmms,
		map<string, vector<int*>> &boundaries, 
		map<string, double**> &transitionCosts);

	static double updateBoundaryAndGetCost(map<string, GuassianModel**> &models, 
		map<string, vector<double**>*> &trainingData, 
		map<string, vector<int>*> &frameLengths,
		map<string, vector<int*>> &boundaries,
		map<string, double**> &transitionCosts);

	static void fillTransitionCostTable(map<string, HmmModel*> &hmms, map<string, double**> &transitionCosts);

public:
	//SegKmeans(double ***_templates, int *_tFrameNum, int templateNum);
	SegKmeans();
	static void segmentalKmeans(map<string, HmmModel*> &hmms, 
		map<string, vector<double**>*> &trainingData, 
		map<string, vector<int>*> &frameLengths);
};

#endif