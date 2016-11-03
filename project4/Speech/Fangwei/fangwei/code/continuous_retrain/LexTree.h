#ifndef _LEXTREE_H_
#define _LEXTREE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

typedef struct _LexNode {
	string word;
	int layer;
	vector<_LexNode*> children;
	bool end;

	_LexNode(string _word, bool _end, int _layer) {
		word = _word;
		end = _end;
		layer = _layer;
	}
} LexNode;

class LexTree {
private:
	void constructTree(LexNode *root, int layerNum);
public:
	LexNode* root;
	LexTree();
};

#endif