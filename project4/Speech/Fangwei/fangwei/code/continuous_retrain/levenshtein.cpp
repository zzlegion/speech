#include "levenshtein.h"

levenshtein::levenshtein() {
	trellis = NULL;
	path = NULL;
}

double levenshtein::getCost(double *a, double *b) {
	double result = 0;
	for (int i = 0; i < FEATURE_LENGTH; i++) {
		result += ((a[i] - b[i]) * (a[i] - b[i]));
	}
	return result;
}

double levenshtein::getCostOnMeanAnVariance(double *mean, double *variance, double *input) {
	double result = 0;
	for (int i = 0; i < FEATURE_LENGTH; i++) {
		//printf("%f %f %f %f\n", ((input[i] - mean[i]) * (input[i] - mean[i]) / (variance[i] * variance[i])), input[i], mean[i], variance[i]);
		result += ((input[i] - mean[i]) * (input[i] - mean[i]) / variance[i]);
		result += log(2 * M_PI * variance[i]);
	}
	result *= 0.5f;
	return result;
}

/*int levenshtein::getThreshold(real ***trellis, int templateNum, int *rowNumber, int colIndex, int method) {
	switch (method) {
		case 0:
			return MAX_INT;
		case 1:
			return ABSOLUTE_THRESHOLD;
		case 2:
			int min = MAX_INT;
			for (int m = 0; m < templateNum; m++) {
				for (int i = 0; i < rowNumber[m]; i++) {
					if (trellis[m][i][colIndex] >= 0 && trellis[m][i][colIndex] < min)
						min = trellis[m][i][colIndex];
				}
			}
			return min + BEAM_WIDTH;
	}
}*/

void levenshtein::initTrellis(int templateNum, int *rows, int cols) {
	if (trellis != NULL) {
		for (int m = 0; m < templateNum; m++) {
			for (int i = 0; i < rows[m]; i++) {
				free(trellis[m][i]);
			}
			free(trellis[m]);
		}
		free(trellis);
		trellis = NULL;
	}
	trellis = (real***)malloc(templateNum * sizeof(real**));
	for (int m = 0; m < templateNum; m++) {
		trellis[m] = (real**)malloc(rows[m] * sizeof(real*));
		for (int i = 0; i < rows[m]; i++) {
			trellis[m][i] = (real*)malloc(cols * sizeof(real));
			for (int k = 0; k < cols; k++)
				trellis[m][i][k] = -1.0f;
			//memset(trellis[m][i], -1.0f, cols * sizeof(real));
		}
		trellis[m][0][0] = 0;
	}
}

void levenshtein::initPath(int templateNum, int *rows, int cols) {
	if (path != NULL) {
		for (int m = 0; m < templateNum; m++) {
			for (int i = 0; i < rows[m]; i++) {
				free(path[m][i]);
			}
			free(path[m]);
		}
		free(path);
		path = NULL;
	}
	path = (pathNode***)malloc(templateNum * sizeof(pathNode**));
	for (int m = 0; m < templateNum; m++) {
		path[m] = (pathNode**)malloc(rows[m] * sizeof(pathNode*));
		for (int i =0; i < rows[m]; i++) {
			path[m][i] = (pathNode*)malloc(cols * sizeof(pathNode));
		}
		path[m][0][0].operation = 1;
		path[m][0][0].row = INVALIDE_FLAG;
		path[m][0][0].col = INVALIDE_FLAG;
	}
}

int levenshtein::getMinDistance(double ***mTemplate, int templateNum, int *tFrameNum, double **mInput, int iFrameNum) {
	int *rows = (int*)malloc(templateNum * sizeof(int));
	memcpy(rows, tFrameNum, templateNum * sizeof(int));
	for (int i = 0; i < templateNum; i++) {
		rows[i] = rows[i];
	}
	int cols = iFrameNum;

	bool *valid = (bool*)malloc(templateNum * sizeof(bool));
	for (int i = 0; i < templateNum; i++) {
		valid[i] = true;
	}

	initPath(templateNum, rows, cols);
	initTrellis(templateNum, rows, cols);

	double threshold = 0;
	for (int j = 0; j < cols - 1; j++) {
		double colMin = MAX_INT;
		for (int m = 0; m < templateNum; m++) {
			if (!valid[m])
				continue;
			valid[m] = false;
			for (int i = 0; i < rows[m]; i++) {
				if (trellis[m][i][j] < 0)// || trellis[m][i][j] > threshold)
					continue;
				valid[m] = true;
				
				for (int z = 0; z < MAX_SKIP_RANGE; z++) {
					int targetRow = i + z;
					if (targetRow >= rows[m])
						break;
					real back = trellis[m][targetRow][j] >= 0 ? trellis[m][targetRow][j] : MAX_INT;
					real backBelow = (targetRow - 1 >= 0 ? (trellis[m][targetRow - 1][j] >= 0 ? trellis[m][targetRow - 1][j] : MAX_INT) : MAX_INT);
					real backBelowBelow = (targetRow - 2 >= 0? (trellis[m][targetRow - 2][j] >= 0 ? trellis[m][targetRow - 2][j] : MAX_INT) : MAX_INT);
					real cost = getCost(mTemplate[m][targetRow], mInput[j + 1]);
					if (backBelow <= back && backBelow <= backBelowBelow) {
						trellis[m][targetRow][j + 1] = backBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 1;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 1;
					}
					else if (back <= backBelow && back <= backBelowBelow) {
						trellis[m][targetRow][j + 1] = back + cost;
						path[m][targetRow][j + 1].row = targetRow;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 0;
					}
					else if (backBelowBelow <= back && backBelowBelow <= backBelow) {
						trellis[m][targetRow][j + 1] = backBelowBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 2;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 2;
					}
					if (trellis[m][targetRow][j + 1] < colMin)
						colMin = trellis[m][targetRow][j + 1];
				}
			}
		}
		threshold = colMin + BEAM_WIDTH;
	}
	
	/*for (int m = 0; m < templateNum; m++) {
		printf("template: %s\n", &mTemplate[m][1]);
		printTrellis(m, mTemplate[m], mInput);
		printf("\n");
	}*/

	double min = MAX_INT;
	int index = -1;
	for (int m = 0; m < templateNum; m++) {
		if (trellis[m][rows[m] - 1][cols - 1] >= 0 && min > trellis[m][rows[m] - 1][cols - 1]) {
			min = trellis[m][rows[m] - 1][cols - 1];
			index = m;
		}
	}

	return index;

	/*if (index >= 0) {
		printf("the best matched template: %s\n", &mTemplate[index][1]);
		printTrellis(index, mTemplate[index], mInput);
		printf("minimun levenshtein distance:%d\n", min);
	} else {
		printf("no best template\n");
	}*/
}

int levenshtein::getMinDistance(GuassianModel ***models, double ***transitionCost, int templateNum, int *tFrameNum, double **mInput, int iFrameNum) {
	int *rows = (int*)malloc(templateNum * sizeof(int));
	memcpy(rows, tFrameNum, templateNum * sizeof(int));
	int cols = iFrameNum;
	
	bool *valid = (bool*)malloc(templateNum * sizeof(bool));
	for (int i = 0; i < templateNum; i++) {
		valid[i] = true;
	}

	/*int cols = iFrameNum;
	int *rows = (int*)malloc(templateNum * sizeof(int));
	for (int i = 0; i < templateNum; i++)
		rows[i] = STATE_NUM;*/

	initPath(templateNum, rows, cols);
	initTrellis(templateNum, rows, cols);

	for (int m = 0; m < templateNum; m++)
		trellis[m][0][0] = models[m][0]->getCost(mInput[0]);

	double threshold = 0;
	for (int j = 0; j < cols - 1; j++) {
		double colMin = MAX_INT;
		for (int m = 0; m < templateNum; m++) {
			if (!valid[m])
				continue;
			valid[m] = false;
			for (int i = 0; i < rows[m]; i++) {
				if (trellis[m][i][j] < 0) // || trellis[m][i][j] > threshold
					continue;
				valid[m] = true;
				
				for (int z = 0; z < MAX_SKIP_RANGE - 1; z++) {
					int targetRow = i + z;
					if (targetRow >= rows[m])
						break;
					real back = trellis[m][targetRow][j] >= 0 ? trellis[m][targetRow][j] : MAX_INT;
					real backBelow = (targetRow - 1 >= 0 ? (trellis[m][targetRow - 1][j] >= 0 ? trellis[m][targetRow - 1][j] : MAX_INT) : MAX_INT);
					back += transitionCost[m][targetRow][targetRow];
					backBelow += (targetRow - 1 >= 0 ? transitionCost[m][targetRow - 1][targetRow] : MAX_INT);
					//real backBelowBelow = (targetRow - 2 >= 0? (trellis[m][targetRow - 2][j] >= 0 ? trellis[m][targetRow - 2][j] : MAX_INT) : MAX_INT);
					real cost = models[m][targetRow]->getCost(mInput[j + 1]);
					if (backBelow <= back) { // && backBelow <= backBelowBelow
						trellis[m][targetRow][j + 1] = backBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 1;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 1;
					}
					else if (back <= backBelow) { // && back <= backBelowBelow
						trellis[m][targetRow][j + 1] = back + cost;
						path[m][targetRow][j + 1].row = targetRow;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 0;
					}
					/*else if (backBelowBelow <= back && backBelowBelow <= backBelow) {
						trellis[m][targetRow][j + 1] = backBelowBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 2;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 2;
					}*/
					if (trellis[m][targetRow][j + 1] < colMin)
						colMin = trellis[m][targetRow][j + 1];
				}
			}
		}
		threshold = colMin + BEAM_WIDTH;
	}

	double min = MAX_INT;
	int index = -1;
	for (int m = 0; m < templateNum; m++) {
		if (trellis[m][rows[m] - 1][cols - 1] >= 0 && min > trellis[m][rows[m] - 1][cols - 1]) {
			min = trellis[m][rows[m] - 1][cols - 1];
			index = m;
		}
	}

	return index;
}

double levenshtein::fillTrellisForKmeansAndReturnCost(GuassianModel **model, double **transitionCost, int tFrameNum, double **mInput, int iFrameNum) {
	int templateNum = 1;
	int *rows = (int*)malloc(templateNum * sizeof(int));
	rows[0] = tFrameNum;
	int cols = iFrameNum;

	initPath(templateNum, rows, cols);
	initTrellis(templateNum, rows, cols);

	for (int m = 0; m < templateNum; m++)
		trellis[m][0][0] = model[0]->getCost(mInput[0]);

	for (int j = 0; j < cols - 1; j++) {
		for (int m = 0; m < templateNum; m++) {
			for (int i = 0; i < rows[m]; i++) {
				if (trellis[m][i][j] < 0)
					continue;
				
				for (int z = 0; z < MAX_SKIP_RANGE - 1; z++) {
					int targetRow = i + z;
					if (targetRow >= rows[m])
						break;
					real back = trellis[m][targetRow][j] >= 0 ? trellis[m][targetRow][j] : MAX_INT;
					real backBelow = (targetRow - 1 >= 0 ? (trellis[m][targetRow - 1][j] >= 0 ? trellis[m][targetRow - 1][j] : MAX_INT) : MAX_INT);
					back += transitionCost[targetRow][targetRow];
					backBelow += (targetRow - 1 >= 0 ? transitionCost[targetRow - 1][targetRow] : MAX_INT);
					//real backBelowBelow = (targetRow - 2 >= 0? (trellis[m][targetRow - 2][j] >= 0 ? trellis[m][targetRow - 2][j] : MAX_INT) : MAX_INT);
					real cost = model[targetRow]->getCost(mInput[j + 1]);
					if (backBelow <= back) { // && backBelow <= backBelowBelow
						trellis[m][targetRow][j + 1] = backBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 1;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 1;
					}
					else if (back <= backBelow) { // && back <= backBelowBelow
						trellis[m][targetRow][j + 1] = back + cost;
						path[m][targetRow][j + 1].row = targetRow;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 0;
					}
					/*else if (backBelowBelow <= back && backBelowBelow <= backBelow) {
						trellis[m][targetRow][j + 1] = backBelowBelow + cost;
						path[m][targetRow][j + 1].row = targetRow - 2;
						path[m][targetRow][j + 1].col = j;
						path[m][targetRow][j + 1].operation = 2;
					}*/
				}
			}
		}
	}
	
	return trellis[0][tFrameNum - 1][iFrameNum - 1];
}

int** levenshtein::getAlignment(int index, int rows, int cols, int *pathLength) {
	if (trellis[index][rows - 1][cols - 1] < 0) {
		printf("no path left.\n");
		return NULL;
	} else {
		*pathLength = cols;
		int **result = (int**)malloc(cols * sizeof(int*));
		const int pointDim = 2;
		for (int i = 0; i < cols; i++) {
			result[i] = (int*)malloc(pointDim * sizeof(int));
			if (result[i] == NULL)
				printf("null!");
		}
		int scan = 0;
		result[scan][0] = rows - 1;
		result[scan][1] = cols - 1;
		scan++;
		pathNode temp = path[index][rows - 1][cols - 1];
		for (int i = 0; i < cols - 1; i++) {
			result[scan][0] = temp.row;
			result[scan][1] = temp.col;
			scan++;
			temp = path[index][temp.row][temp.col];
		}
		return result;
	}
}

