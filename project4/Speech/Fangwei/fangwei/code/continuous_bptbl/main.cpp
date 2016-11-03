#include <iostream>
#include "recorder.h"

int main() {
	string modelFile = "model_continuous_large.dat";
	string silFile = "model_continuous_sil_large.dat";
	bool loopBack = true; // false for p1 
	int beamWidth = 15;
	//Recognizer* checker = new Recognizer(loopBack, beamWidth, modelFile);
	//mQueue nodeHolder;
	Recorder::record(loopBack, beamWidth, modelFile, silFile);
	
	return 0;
}