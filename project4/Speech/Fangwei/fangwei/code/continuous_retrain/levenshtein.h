#ifndef _LEVENSHTEIN_H_
#define _LEVENSHTEIN_H_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "SegKmeans.h"
#include "GuassianModel.h"

#define MAX_INT (1e10)
#define ABSOLUTE_THRESHOLD (3)
#define BEAM_WIDTH (100000)
#define INVALIDE_FLAG (-1)
#define MAX_SKIP_RANGE (3)
#define PREFIX_INT (10)
#define PREFIX_double (0.001f)

typedef double real;

typedef struct pathNode_ {
	int row;
	int col;
	int operation;  // 0 - horizontal; 1 - diag; 2 - skip
} pathNode;

class levenshtein {
private:
	real ***trellis;
	pathNode ***path;
	void initTrellis(int templateNum, int *rows, int cols);
	void initPath(int templateNum, int *rows, int cols);
	double getCost(double *a, double *b);
	double getCostOnMeanAnVariance(double *mean, double *variance, double *input);

public:
	levenshtein();
	int getMinDistance(double ***mTemplate, int templateNum, int *tFrameNum, double **mInput, int iFrameNum);
	int getMinDistance(GuassianModel ***models, double ***transitionCost, int templateNum, int *tFrameNum, double **mInput_, int iFrameNum);
	double fillTrellisForKmeansAndReturnCost(GuassianModel **model, double **transitionCost, int tFrameNum, double **mInput_, int iFrameNum);
	int** getAlignment(int index, int rows, int cols, int *pathLength);
};

#endif