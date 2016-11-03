#include <iostream>
#include "Trainer.h"

int main() {
	string modelFile = "model.dat";
	string silFile = "model_sil.dat";
	bool loopBack = true; // false for p1 
	int beamWidth = 15;
	Trainer* trainer = new Trainer(loopBack, beamWidth, modelFile, silFile);

	string dir = "record/";
	string temp[6] = {"0123456789", "0987654321", "1234567890", "1357902468", "8642097531", "9876543210"};
	vector<string> sequences(&temp[0], &temp[0] + 6);
	string temp_[5] = {"/1.wav", "/2.wav", "/3.wav", "/4.wav", "/5.wav"};
	vector<string> names(&temp_[0], &temp_[0] + 5);
	trainer->loadTrainingData(dir, sequences, names);
	trainer->trainModel();

	modelFile = "model_continuous.dat";
	silFile = "model_continuous_sil.dat";
	trainer->saveModel(modelFile, silFile);
	
	return 0;
}