#include "LexTree.h"

LexTree::LexTree() {
	string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "sil", "nil"};
	LexNode *pass = new LexNode(names[11], false, 1);
	constructTree(pass, 4);

	root = new LexNode(names[11], false, 0);
	LexNode *nil2 = new LexNode(names[11], false, 2);
	for (int i = 2; i <= 9; i++) {
		LexNode *node = new LexNode(names[i], false, 1);
		node->children.push_back(nil2);
		root->children.push_back(node);
	}
	root->children.push_back(pass);
	constructTree(nil2, 6);
}

void LexTree::constructTree(LexNode *root, int layerNum) {
	string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "sil", "nil"};
	LexNode* currentNode = root;
	for (int i = 0; i < layerNum; i++) {
		bool end = (layerNum - 1 == i ? true : false);
		LexNode *nil = new LexNode(names[11], end, currentNode->layer + 2);
		for (int j = 0; j <= 9; j++) {
			LexNode *node = new LexNode(names[j], false, currentNode->layer + 1);
			node->children.push_back(nil);
			currentNode->children.push_back(node);
		}
		currentNode = nil;
	}
}