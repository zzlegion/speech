#include <stdio.h>
#include "DtwTester.h"

DtwTester::DtwTester(int type) {
	const int digitNum = 10;
	const int dirNum = 7;
	//char recordFiles[digitNum][MAX_STRING_LENGTH] = {"two", "three", "six"};
	char recordFiles[digitNum][MAX_STRING_LENGTH] = {"zero", "one", "two", "three", "four", "five", 
		"six", "seven", "eight", "nine"};
	char dir[dirNum][MAX_STRING_LENGTH] = {"wei/", "weif/", "gaoyuan/", "pengzhou/", "dajian1/", "fengwei1/", "chenzhun1/"};

	mTemplate = NULL;
	labels = NULL;
	templateNum = 0;
	tFrameNum = NULL;
	/*means = NULL;
	variances = NULL;*/
	models = NULL;

	switch(type) {
	case 0:
		loadTemplates(digitNum, dirNum, recordFiles, dir);
		break;
	case 1:
		loadTemplatesKmeans(digitNum, dirNum, recordFiles, dir);
		break;
	}
}

void DtwTester::loadTemplates(int digitNum, int dirNum, char recordFiles[][MAX_STRING_LENGTH], char dir[][MAX_STRING_LENGTH]) {
	//char filePath[MAX_STRING_LENGTH] = "C:\\Users\\acer\\Desktop\\speech\\ASR_Dataset\\all";//"F:\\code\\endpointing\\endpointing\\wei";  
	//vector<string> files; 
	//getAllFiles(filePath, files);
	//
	//mTemplate = (double***)malloc(files.size() * sizeof(double**));
	//labels = (char**)malloc(files.size() * sizeof(char*));
	//for (int i = 0; i < files.size(); i++) {
	//	labels[i] = (char*)malloc(MAX_STRING_LENGTH * sizeof(char));
	//}
	//templateNum = files.size();
	//tFrameNum = (int*)malloc(files.size() * sizeof(int));

	//for (int i = 0; i < files.size(); i++) {
	//	char filename[MAX_STRING_LENGTH];
	//	strcpy(filename, filePath);
	//	strcat(filename, "\\");
	//	strcat(filename, files[i].c_str());
	//	strcpy(labels[i], files[i].substr(0, 3).c_str());
	//	MfccExtractor::getMfccFromWav(filename, &tFrameNum[i], &mTemplate[i]);
	//}
	
	mTemplate = (double***)malloc(digitNum * dirNum * sizeof(double**));
	labels = (char**)malloc(digitNum * dirNum * sizeof(char*));
	for (int i = 0; i < digitNum * dirNum; i++) {
		labels[i] = (char*)malloc(MAX_STRING_LENGTH * sizeof(char));
	}
	templateNum = digitNum * dirNum;
	tFrameNum = (int*)malloc(digitNum * dirNum * sizeof(int));

	int scan = 0;
	for (int i = 0; i < dirNum; i++) {
		for (int j = 0; j < digitNum; j++) {
			char filename[MAX_STRING_LENGTH];
			strcpy(filename, dir[i]);
			strcat(filename, recordFiles[j]);
			strcat(filename, ".wav");
			labels[scan] = (char*)malloc(strlen(recordFiles[j]) * sizeof(char));
			strcpy(labels[scan], recordFiles[j]);
			//labels[scan] = recordFiles[j];
			MfccExtractor::getMfccFromWav(filename, &tFrameNum[scan], &mTemplate[scan]);
			/*for (int k = 0; k < 3* DCT_PRINCIPAL_NUM; k++) {
				printf("%f\n", mTemplate[scan][10][k]);
			}*/
			scan++;
		}
	}
}

void DtwTester::loadTemplatesKmeans(int digitNum, int dirNum, char recordFiles[][MAX_STRING_LENGTH], char dir[][MAX_STRING_LENGTH]) {
	/*means = (double***)malloc(digitNum * sizeof(double**));
	for (int i = 0; i < digitNum; i++) {
		means[i] = (double**)malloc(STATE_NUM * sizeof(double*));
		for (int j = 0; j < STATE_NUM; j++) {
			means[i][j] = (double*)malloc(FEATURE_LENGTH * sizeof(double));
		}
	}

	variances = (double***)malloc(digitNum * sizeof(double**));
	for (int i = 0; i < digitNum; i++) {
		variances[i] = (double**)malloc(STATE_NUM * sizeof(double*));
		for (int j = 0; j < STATE_NUM; j++) {
			variances[i][j] = (double*)malloc(FEATURE_LENGTH * sizeof(double));
		}
	}*/

	models = (GuassianModel***)malloc(digitNum * sizeof(GuassianModel**));
	for (int i = 0; i < digitNum; i++) {
		models[i] = (GuassianModel**)malloc(STATE_NUM * sizeof(GuassianModel*));
		for (int j = 0; j < STATE_NUM; j++) {
			models[i][j] = new GuassianModel(GUASSIAN_NUM);
		}
	}

	printf("size: %d\n", sizeof(GuassianModel*));

	transitionCost = (double***)malloc(digitNum * sizeof(double**));
	for (int i = 0; i < digitNum; i++) {
		transitionCost[i] = (double**)malloc(STATE_NUM * sizeof(double*));
		for (int j = 0; j < STATE_NUM; j++) {
			transitionCost[i][j] = (double*)malloc(STATE_NUM * sizeof(double));
		}
	}

	labels = (char**)malloc(digitNum * sizeof(char*));
	for (int i = 0; i < digitNum ; i++) {
		labels[i] = (char*)malloc(MAX_STRING_LENGTH * sizeof(char));
	}
	templateNum = digitNum;

	for (int i = 0; i < digitNum; i++) {
		//labels[i] = (char*)malloc(MAX_STRING_LENGTH * sizeof(char));
		strcpy(labels[i], recordFiles[i]);
		double ***temp = (double***)malloc(dirNum * sizeof(double**));
		int *frameNum = (int*)malloc(dirNum * sizeof(int));
		for (int j = 0; j < dirNum; j++) {
			char filename[MAX_STRING_LENGTH];
			strcpy(filename, dir[j]);
			strcat(filename, recordFiles[i]);
			strcat(filename, ".wav");
			MfccExtractor::getMfccFromWav(filename, &frameNum[j], &temp[j]);
		}
		printf("label: %s\n", labels[i]);
		SegKmeans *sk = new SegKmeans(temp, frameNum, dirNum);
		sk->kmeans(models[i], transitionCost[i]);
		free(temp);
		free(frameNum);
		delete(sk);
	}

	tFrameNum = (int*)malloc(digitNum * sizeof(int));
	for (int i = 0; i < digitNum; i++)
		tFrameNum[i] = STATE_NUM;
}

void DtwTester::test(int type) {
	switch (type) {
	case 0:
		Recorder::record(mTemplate, templateNum, tFrameNum, labels, type, NULL, NULL);
		break;
	case 1:
		Recorder::record(NULL, templateNum, tFrameNum, labels, type, models, transitionCost);
		break;
	}
}

void DtwTester::batchEvaluation(int type) {
	int totalFileNum = 0;
	int correctFileNum = 0;

	char filePath[MAX_STRING_LENGTH] = "C:\\Users\\acer\\Desktop\\speech\\ASR_Dataset\\testing";
	vector<string> files; 
	DtwTester *tester = new DtwTester(type);  // 0 - dtw, 1 - kmeans, 2 - gmm 
	tester->getAllFiles(filePath, files);

	for (int i = 0; i < files.size(); i++) {
		int iFrameNum = 0;
		double **mInput = NULL;
		char temp[MAX_STRING_LENGTH] = "";
		strcat(temp, filePath);
		strcat(temp, "\\");
		strcat(temp, files[i].c_str());
		MfccExtractor::getMfccFromWav(temp, &iFrameNum, &mInput);
		levenshtein *lev = new levenshtein();
		int index = -1;
		switch (type) {
		case 0:
			index = lev->getMinDistance(mTemplate, templateNum, tFrameNum, mInput, iFrameNum);
			break;
		case 1:
			index = lev->getMinDistance(models, transitionCost, templateNum, tFrameNum, mInput, iFrameNum);
		}
		delete(lev);
		totalFileNum++;
		if (index != -1 && labels[index][0] == files[i].c_str()[0] && labels[index][1] == files[i].c_str()[1] 
				&& labels[index][2] == files[i].c_str()[2]) {
			correctFileNum++;
		}
	}

	printf("accuracy: %f\n", (float)correctFileNum / totalFileNum);
}

void DtwTester::getAllFiles(string path, vector<string>& files) {   
    long   hFile   =   0;  
    struct _finddata_t fileinfo; 
    string p; 
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {   
            if((fileinfo.attrib &  _A_SUBDIR))
            {  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
                {
                    //files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					//files.push_back(fileinfo.name);
                    getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
                }
            }  
            else  
            {  
                //files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
				files.push_back(fileinfo.name);
            }  
        }while(_findnext(hFile, &fileinfo)  == 0);
        _findclose(hFile); 
    } 
}

void DtwTester::saveModel(string filename) {
	ofstream out(filename);
	if (out.is_open()) {
		out << DIGIT_NUM << endl;
		out << STATE_NUM << endl;
		out << GUASSIAN_NUM << endl;
		out << FEATURE_LENGTH << endl;

		for (int i = 0; i < DIGIT_NUM; i++) {
			out << endl << endl;
			for (int j = 0; j < STATE_NUM; j++) {
				for (int m = 0; m < GUASSIAN_NUM; m++) {
					out << models[i][j]->alpha[m] << endl;
					for (int n = 0; n < FEATURE_LENGTH; n++) {
						out << models[i][j]->means[m][n] << " ";
					}
					out << endl;
					for (int n = 0; n < FEATURE_LENGTH; n++) {
						out << models[i][j]->variances[m][n] << " ";
					}
					out << endl;
				}
				out << endl;
			}
		}
	} else {
		cout << "unable to open file" << endl;
	}
	out.flush();
	out.close();
}