#include <stdio.h>
#include "stdlib.h"
#include <math.h>
#include <memory>
#include <vector>
#include <iostream>
#include <fstream>
#include <io.h>
#include <sstream>

#include "getFile.h"
#include "dtwSeg.h"
#include "segment.h"


using namespace std;

vector<int> calDigitIdx(string &fileName);


////////////////////////////////////////* Consider zero pronunciation Z(0) and O(11), silence(10) */

int main()
{
	//char * filePath = "E:\\SpeechReconition\\hw5\\hwdata_assign6\\train_mfcc";
	//char * filePath = "E:\\Train\\mfcc\\mfcc\\coutinuous_mfcc_long";
	//char * filePath = "E:\\Train\\mfcc\\mfcc\\coutinuous_mfcc_short";

	char * filePath = "C:\\Users\\june\\Desktop\\project7_new\\train_material";
	//char * filePath = "C:\\Users\\june\\Desktop\\hwdata_assign6\\train_mfcc_debug";

	vector<string> files;
	getFiles(filePath, "txt", files);   //get the training file names

	cout<<files.size()<<endl;

	vector<vector<vector<float> > > mfccData(files.size());   	//mfcc data of all input audio
	vector<vector<int> > digitSequence(files.size());   	//ground truth of digit sequence of input audio

	for(int n=0;n<files.size();n++)
	{
		//cout<<files[n]<<endl;
		vector<int> digitGT = calDigitIdx(files[n]);
		ifstream infile(files[n]);
		string line;
		vector<vector<float> > temp1;
		while(getline(infile, line))
		{
			vector<float> temp2;
			istringstream is(line);
			string word;
			while(is>>word)
			{
				temp2.push_back(atof(word.data()));
			}
			temp1.push_back(temp2);
		}
		infile.close();
		mfccData[n]=temp1;
		digitSequence[n] = digitGT;
	}
	
	cout<< "There are: "<<mfccData.size()<<"training files"<< endl;

	segment(mfccData, digitSequence);
	
	system("pause");
    return 0;
    
}


vector<int> calDigitIdx(string &fileName)
{
	// zero(Z)(0),  silence(10),  zero(O)(11)
	cout << fileName << endl;

	std::size_t found1 = fileName.find_last_of("\\");
	std::size_t found2 = fileName.find_last_of(".");
	string name = fileName.substr(found1+1,found2-found1-1);

	vector<int> digitIdx;
	digitIdx.push_back(10); //silence at the beginning
	for(int i=4;i<name.size()-1;i++)  // fileName format: FBH_8O39A
	{
		if(isdigit(name[i]))
			digitIdx.push_back(atoi(name.substr(i,1).c_str()));
		else if (name[i]=='Z')
			digitIdx.push_back(0);
		else if(name[i]=='O')
			digitIdx.push_back(11);   
		else
			cout<<"wrong fileName format"<<endl;
	}
	return digitIdx;
}



