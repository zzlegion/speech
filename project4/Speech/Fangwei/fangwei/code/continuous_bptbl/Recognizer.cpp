#include "Recognizer.h"

mQueue::mQueue() {
	head = 0;
	tail = 0;
	_queue = (TrellisNode**)malloc(100000 * sizeof(TrellisNode*));
}

TrellisNode* mQueue::front() {
	return _queue[head];
}

void mQueue::pop() {
	head++;
}

void mQueue::push(TrellisNode *node) {
	if (tail - head > 0 && node->name == _queue[tail - 1]->name && node->layer == _queue[tail - 1]->layer)
		return;
	_queue[tail] = node;
	tail++;
}

int mQueue::size() {
	return tail - head;
}

void mQueue::reset() {
	head = 0;
	tail = 0;
}


Recognizer::Recognizer(bool _loopBack, int _beamWidth, string modelFile, string silFile) {
	loopBack = _loopBack;
	beamWidth = _beamWidth;
	loadHmms(modelFile, "nonSil");
	loadHmms(silFile, "sil");
}

void Recognizer::loadHmms(string modelFile, string type) {
	string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "o", "sil"};
	if (type == "sil")
		names[0] = "sil";
	ifstream in(modelFile);
	if (in.is_open()) {
		int digitNum;
		int stateNum;
		int guassianNum;
		int featureNum;
		in >> digitNum >> stateNum >> guassianNum >> featureNum;
		double ***means = (double***)malloc(stateNum * sizeof(double**));
		double ***variances = (double***)malloc(stateNum * sizeof(double**));
		double **alpha = (double**)malloc(stateNum * sizeof(double*));
		for (int i = 0; i < stateNum; i++) {
			means[i] = (double**)malloc(guassianNum * sizeof(double*));
			variances[i] = (double**)malloc(guassianNum * sizeof(double*));
			alpha[i] = (double*)malloc(guassianNum * sizeof(double));
			for (int j = 0; j < guassianNum; j++) {
				means[i][j] = (double*)malloc(featureNum * sizeof(double));
				variances[i][j] = (double*)malloc(featureNum * sizeof(double));
			}
		}

		double **transitionCost = (double**)malloc(stateNum * sizeof(double*));
		for (int i = 0; i < stateNum; i++) {
			transitionCost[i] = (double*)malloc(stateNum * sizeof(double));
		}

		for (int i = 0; i < digitNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				for (int m = 0; m < guassianNum; m++) {
					in >> alpha[j][m];
					for (int n = 0; n < featureNum; n++) {
						in >> means[j][m][n];
					}
					for (int n = 0; n < featureNum; n++) {
						in >> variances[j][m][n];
					}
				}			
			}
			for (int m = 0; m < stateNum; m++) {
				for (int n = 0; n < stateNum; n++) {
					in >> transitionCost[m][n];
				}
			}
			HmmModel *model = new HmmModel(means, variances, alpha, transitionCost, guassianNum, stateNum, featureNum);
			hmms[names[i]] = model;
		}

		// free
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < guassianNum; j++) {
				free(means[i][j]);
				free(variances[i][j]);
			}
			free(means[i]);
			free(variances[i]);
			free(alpha[i]);
		}
		free(means);
		free(variances);
		free(alpha);

		for (int i = 0; i < stateNum; i++) {
			free(transitionCost[i]);
		}
		free(transitionCost);
	}
}

TrellisNode* Recognizer::constructTree(TrellisNode *root, int layerNum, int trellisLength) {
	string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "sil", "nil"};
	TrellisNode *currentNode = root;
	for (int i = 0; i < layerNum; i++) {
		bool end = (layerNum - 1 == i ? true : false);
		TrellisNode *nil = new TrellisNode("nil", end, trellisLength, NULL, -1, 1, currentNode->layer + 2);
		for (int j = 0; j <= 9; j++) {
			TrellisNode *node = new TrellisNode(names[j], false, trellisLength, NULL, -1, hmms[names[j]]->stateNum, currentNode->layer + 1);
			node->children.push_back(nil);
			currentNode->children.push_back(node);
		}
		currentNode = nil;
	}
	return currentNode;
}

double Recognizer::getCost(string &name, int stateIndex, double *input, int from, int to) {
	double transCost = 0;
	if (from >= 0 && to >= 0) {
		transCost = hmms[name]->transitionCost[from][to];
	} else {
		double penalty = 40.0f; // 200 is ok for complete grammar
		transCost = penalty;
	}
	double nodeCost = hmms[name]->guassianModels[stateIndex]->getCost(input);
	return nodeCost + transCost;
}

string Recognizer::recognize(double **input, int inputLength, mQueue &nodeHolder) {
	int trellisLength = inputLength;
	vector<TrellisNode*> leafs;
	TrellisNode *root = NULL;
	string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "sil"}; // "o", 
	//-------------------------------------------------------------------------------------------------------------
	// init tree
	if (loopBack) {
		root = new TrellisNode("nil", true, trellisLength, NULL, -1, 1, 0);
		//TrellisNode *nil2 = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 2);
		for (int i = 0; i <= 10; i++) {
			TrellisNode *node = new TrellisNode(names[i], false, trellisLength, NULL, -1, hmms[names[i]]->stateNum, 1);
			node->trellis[0][0] = getCost(node->name, 0, input[0], -1, -1);
			node->children.push_back(root);
			root->children.push_back(node);
		}
		//root->children.push_back(nil2);
		leafs.push_back(root);
	} else {
		TrellisNode *pass = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 1);
		TrellisNode *nil3 = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 3);
		for (int i = 0; i <= 9; i++) {
			TrellisNode *node = new TrellisNode(names[i], false, trellisLength, NULL, -1, hmms[names[i]]->stateNum, 2);
			node->trellis[0][0] = getCost(node->name, 0, input[0], -1, -1);
			node->children.push_back(nil3);
			pass->children.push_back(node);
		}
		leafs.push_back(constructTree(nil3, 3, trellisLength));

		root = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 0);
		TrellisNode *nil2 = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 2);
		for (int i = 0; i <= 9; i++) {
			TrellisNode *node = new TrellisNode(names[i], false, trellisLength, NULL, -1, hmms[names[i]]->stateNum, 1);
			node->trellis[0][0] = getCost(node->name, 0, input[0], -1, -1);
			node->children.push_back(nil2);
			root->children.push_back(node);
		}
		root->children.push_back(pass);
		leafs.push_back(constructTree(nil2, 6, trellisLength));  //here
		//leafs.push_back(nil2);

		//TrellisNode *root = new TrellisNode("nil", false, trellisLength, NULL, -1, 1, 0);
		////root->trellis[0][0] = 0;
		//TrellisNode *nil2 = new TrellisNode("nil", true, trellisLength, NULL, -1, 1, 2);
		//for (int i = 1; i <= 9; i++) {
		//	TrellisNode *node = new TrellisNode(names[i], false, trellisLength, NULL, -1, hmms[names[i]]->stateNum, 1);
		//	node->trellis[0][0] = getCost(node->name, 0, input[0], -1, -1);
		//	node->children.push_back(nil2);
		//	root->children.push_back(node);
		//}
		//leafs.push_back(nil2);
	}
	//--------------------------------------------------------------------------------------------------------------

	for (int i = 0; i < trellisLength - 1; i++) {
		double threshold = 100000000000;

		nodeHolder.reset();
		nodeHolder.push(root);
		while(nodeHolder.size() > 0) {
			TrellisNode* node = nodeHolder.front();
			nodeHolder.pop();

			for (int m = 0; m < node->stateNum - 1; m++) {
				if (node->trellis[m][i] < threshold && node->trellis[m][i] >= 0) {
					float insCost = getCost(node->name, m, input[i + 1], m, m);
					if (node->trellis[m][i + 1] < 0 || node->trellis[m][i] + insCost < node->trellis[m][i + 1]) {
						node->trellis[m][i + 1] = node->trellis[m][i] + insCost;
						if (m == 0)
							node->parent[i + 1] = node;
						node->preNodeY[m][i + 1] = m;
						node->preNodeX[m][i + 1] = i;
					}

					float subCost = getCost(node->name, m + 1, input[i + 1], m, m + 1);
					if (node->trellis[m + 1][i + 1] < 0 || node->trellis[m][i] + subCost < node->trellis[m + 1][i + 1]) {
						node->trellis[m + 1][i + 1] = node->trellis[m][i] + subCost;
						//node->parent = node;
						node->preNodeY[m + 1][i + 1] = m;
						node->preNodeX[m + 1][i + 1] = i;
					}
				}
			}

			int topRowIndex = node->stateNum - 1;
			if (node->name != "nil") {
				if (node->trellis[topRowIndex][i] < threshold && node->trellis[topRowIndex][i] >= 0) {
					float insCost = getCost(node->name, topRowIndex, input[i + 1], topRowIndex, topRowIndex);
					if (node->trellis[topRowIndex][i + 1] < 0 || node->trellis[topRowIndex][i] + insCost < node->trellis[topRowIndex][i + 1]) {
						node->trellis[topRowIndex][i + 1] = node->trellis[topRowIndex][i] + insCost;
						//node->parent = node;
						node->preNodeY[topRowIndex][i + 1] = topRowIndex;
						node->preNodeX[topRowIndex][i + 1] = i;
					}
				}
			}
			for (int m = 0; m < node->children.size(); m++) {
				nodeHolder.push(node->children[m]);
				if (node->children[m]->name == "nil") {
					if (node->trellis[topRowIndex][i] < threshold && node->trellis[topRowIndex][i] >= 0) {
						if (node->children[m]->trellis[0][i] < 0 || node->trellis[topRowIndex][i] < node->children[m]->trellis[0][i]) {
							node->children[m]->trellis[0][i] = node->trellis[topRowIndex][i];
							node->children[m]->parent[i] = node;
							node->children[m]->preNodeY[0][i] = node->stateNum - 1;
							node->children[m]->preNodeX[0][i] = i;
						}
					}

					if (node->trellis[topRowIndex][i + 1] < threshold && node->trellis[topRowIndex][i + 1] >= 0) {
						if (node->children[m]->trellis[0][i + 1] < 0 || node->trellis[topRowIndex][i + 1] < node->children[m]->trellis[0][i + 1]) {
							node->children[m]->trellis[0][i + 1] = node->trellis[topRowIndex][i + 1];
							node->children[m]->parent[i + 1] = node;
							node->children[m]->preNodeY[0][i + 1] = node->stateNum - 1;
							node->children[m]->preNodeX[0][i + 1] = i + 1;
						}
					}
				} else {
					if (node->trellis[topRowIndex][i] < threshold && node->trellis[topRowIndex][i] >= 0) {
						float subCost = getCost(node->children[m]->name, 0, input[i + 1], -1, -1);
						if (node->children[m]->trellis[0][i + 1] < 0 || node->trellis[topRowIndex][i] + subCost < node->children[m]->trellis[0][i + 1]) {
							node->children[m]->trellis[0][i + 1] = node->trellis[topRowIndex][i] + subCost;
							node->children[m]->parent[i + 1] = node;
							node->children[m]->preNodeY[0][i + 1] = node->stateNum - 1;
							node->children[m]->preNodeX[0][i + 1] = i;
						}
					}
				}
			}
			if (loopBack && nodeHolder.size() > 0 && node->layer != 0 && nodeHolder.front()->layer == 0)
				break;
		}
	}

	TrellisNode *endNode = leafs[0];
	for (int i = 1; i < leafs.size(); i++) {
		if (leafs[i]->trellis[0][inputLength - 1] < endNode->trellis[0][inputLength - 1]) {
			endNode = leafs[i];
		}
	}

	string result = "";
	if (endNode->end == false) {
		result =  "error";
	} else {
		result = retrieveStringFromEnd(endNode, trellisLength);
	}

	// delete nodes
	nodeHolder.reset();
	nodeHolder.push(root);
	while(nodeHolder.size() > 0) {
		TrellisNode* node = nodeHolder.front();
		nodeHolder.pop();
		for (int i = 0; i < node->children.size(); i++) {
			if (loopBack && node->layer > 0 )
				continue;
			nodeHolder.push(node->children[i]);
		}
		for (int i = 0; i < node->stateNum; i++) {
			free(node->trellis[i]);
			free(node->preNodeX[i]);
			free(node->preNodeY[i]);
		}
		free(node->parent);
		free(node->trellis);
		free(node->preNodeX);
		free(node->preNodeY);
		delete node;
	}

	return result;
}

string Recognizer::retrieveStringFromEnd(TrellisNode *endNode, int trellisLength) {
	/*string names[12] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "sil", "nil"};*/
	string result = "";
	TrellisNode *node = endNode;
	int index = trellisLength - 1;
	while (node->parent[index] != NULL) {
		if (node->parent[index]->name == "nil") {
			int temp = node->preNodeX[0][index];
			node = node->parent[index];
			index = temp;
		} else {
			int tempx = node->preNodeX[0][index];
			int tempy = node->parent[index]->stateNum - 1;
			node = node->parent[index];
			while (true) {
				if (tempy == 0 && node->parent[tempx] != node)
					break;
				int tx = node->preNodeX[tempy][tempx];
				int ty = node->preNodeY[tempy][tempx];
				tempx = tx;
				tempy = ty;
			}
			if (node->name != "sil")
				result.append(node->name);
			index = tempx;
		}
	}
	reverse(result.begin(), result.end());
	return result;
}