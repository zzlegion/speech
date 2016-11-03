#ifndef _DTWTESTER_H_
#define _DTWTESTER_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <io.h>
#include "MfccExtractor.h"
#include "levenshtein.h"
#include "readwave.h"
#include "Recorder.h"
#include "SegKmeans.h"
#include "GuassianModel.h"

using namespace std;

#define DIGIT_NUM (10)
#define MAX_STRING_LENGTH (100)
#define GUASSIAN_NUM (4)

class DtwTester {
private:
	double ***mTemplate;
	char **labels;
	int templateNum;
	int *tFrameNum;

	/*double ***means;
	double ***variances;*/
	double ***transitionCost;

	GuassianModel ***models;

	void loadTemplates(int digitNum, int dirNum, char recordFiles[][MAX_STRING_LENGTH], char dir[][MAX_STRING_LENGTH]);
	void loadTemplatesKmeans(int digitNum, int dirNum, char recordFiles[][MAX_STRING_LENGTH], char dir[][MAX_STRING_LENGTH]);

public:
	DtwTester(int type);
	void test(int type);
	void batchEvaluation(int type);
	static void getAllFiles(string path, vector<string>& files);

	void saveModel(string filename);
};

#endif