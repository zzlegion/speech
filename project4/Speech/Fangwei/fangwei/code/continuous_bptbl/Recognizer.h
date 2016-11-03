#ifndef _RECOGNIZER_H_
#define _RECOGNIZER_H_

#include "HmmModel.h"
#include <map>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
//#include "LexTree.h"

using namespace std;

struct TrellisNode {
	string name;
	int layer;
	int stateNum;

	TrellisNode **parent;
	int selfPosition;

	vector<TrellisNode*> children;

	double **trellis;
	double **preNodeX;
	double **preNodeY;
	bool end;

	TrellisNode(string _word, bool _end, int inputLen, TrellisNode* _parent, int _selfPosition, int _stateNum, int _layer) {
		name = _word;
		layer = _layer;
		stateNum = _stateNum;

		//parent = _parent;
		selfPosition = _selfPosition;
		end = _end;

		parent = (TrellisNode**)malloc(inputLen * sizeof(TrellisNode*));
		for (int i = 0; i < inputLen; i++) {
			parent[i] = NULL;
		}

		trellis = (double**)malloc(stateNum * sizeof(double*));
		preNodeX = (double**)malloc(stateNum * sizeof(double*));
		preNodeY = (double**)malloc(stateNum * sizeof(double*));
		for (int i = 0; i < stateNum; i++) {
			trellis[i] = (double*)malloc(inputLen * sizeof(double));
			preNodeX[i] = (double*)malloc(inputLen * sizeof(double));
			preNodeY[i] = (double*)malloc(inputLen * sizeof(double));
			for (int j = 0; j < inputLen; j++) {
				trellis[i][j] = -1;
				preNodeX[i][j] = -1;
				preNodeY[i][j] = -1;
			}
		}
	}
};

class mQueue{
private:
	int head;
	int tail;
	TrellisNode** _queue;

public:
	mQueue();
	TrellisNode* front();
	void pop();
	void push(TrellisNode *node);
	int size();
	void reset();
};

class Recognizer {
private:
	int beamWidth;
	bool loopBack;
	map<string, HmmModel*> hmms;
	string retrieveStringFromEnd(TrellisNode *endNode, int trellisLength);
	void loadHmms(string modelFile, string type);
	TrellisNode* constructTree(TrellisNode *root, int layerNum, int trellisLength);
	double getCost(string &name, int stateIndex, double *input, int from, int to);

public:
	Recognizer(bool _loopBack, int _beamWidth, string modelFile, string silFile);
	string recognize(double **input, int inputLength, mQueue &nodeHolder);
};

#endif