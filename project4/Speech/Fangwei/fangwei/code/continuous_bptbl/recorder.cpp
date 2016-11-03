#include "recorder.h"

static std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

static void getRecords(string transcriptFile, map<string, vector<string>> &record){
	map<string, string> mapper;
	mapper["one"] = "1";
	mapper["two"] = "2";
	mapper["three"] = "3";
	mapper["four"] = "4";
	mapper["five"] = "5";
	mapper["six"] = "6";
	mapper["seven"] = "7";
	mapper["eight"] = "8";
	mapper["nine"] = "9";
	mapper["zero"] = "0";
	mapper["oh"] = "o";
	ifstream in(transcriptFile);
	string line;
	if (in.is_open()) {
		while (getline(in, line)) {
			vector<string> tokens = split(line, ' ');
			string sequence = "";
			for (int i = 0; i < tokens.size() - 1; i++) {
				sequence.append(mapper[tokens[i]]);
			} 
			string name = tokens[tokens.size()-1];
			name = name.substr(1, name.length() - 2);
			if (record.find(sequence) != record.end()) {
				record[sequence].push_back(name);
			} else {
				vector<string> files;
				files.push_back(name);
				record[sequence] = files;
			}
		}
	}
}


typedef struct
{
    int frameIndex;  /* Index into sample array. */
    int maxFrameIndex;
    SAMPLE *recordedSamples;

	int speechCount;
	int silenceCount;
	bool existSpeech;
	int SpeechStart;
}
paTestData;

int Recorder::recordCallback( const void *inputBuffer, void *outputBuffer,
                           unsigned long framesPerBuffer,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void *userData)
{
    paTestData *data = (paTestData*)userData;
    const SAMPLE *buffer = (const SAMPLE*)inputBuffer;
    SAMPLE *record = &data->recordedSamples[data->frameIndex * NUM_CHANNELS];
    int finished = paContinue;

    (void) outputBuffer; /* Prevent unused variable warnings. */
    (void) timeInfo;
    (void) statusFlags;
    (void) userData;

	if( data->maxFrameIndex - data->frameIndex < framesPerBuffer)
        return paComplete;

	double energy = 0;

    if( inputBuffer == NULL )
    {
        for(int i = 0; i < framesPerBuffer; i++)
        {
            *record++ = SAMPLE_SILENCE;
        }
    }
    else
    {
        for(int i=0; i < framesPerBuffer; i++)
        {
			energy += *buffer * *buffer;
            *record++ = *buffer++;
        }
    }
	energy = log(energy);
	if (energy > OFFSET_THRES) {
		if (data->speechCount == 0)
			data->SpeechStart = data->frameIndex;
		data->speechCount++;
		data->silenceCount = 0;
	} else {
		if (data->speechCount > 0)
			data->silenceCount++;
		//data->speechCount = 0;
	}
	if (data->speechCount >= SHORTEST_SEGMENT) {
		data->existSpeech = true;
	}
    data->frameIndex += framesPerBuffer;
	if (data->existSpeech && data->silenceCount >= SILENCE_LENGTH)
		return paComplete;
	else
		return paContinue;
}

void Recorder::record(bool loopBack, int beamWidth, string modelFile, string silFile) {
	//string modelFile = "model.dat";
	//bool loopBack = false; // false for p1 
	//int beamWidth = 15;
	printf("loading model...");
	Recognizer* recognizer = new Recognizer(loopBack, beamWidth, modelFile, silFile);
	printf("loading done!");
	mQueue nodeHolder;

	//------------
	//string dir = "hwdata/test/";
	//string transcriptFile = "hwdata/TEST.transcripts";
	//map<string, vector<string>> records;
	//getRecords(transcriptFile, records);

	//int iFrameNum = 0;
	//double **mInput = NULL;
	//int count = 0;
	//int correct = 0;

	//for (map<string, vector<string>>::iterator iter = records.begin(); iter != records.end(); iter++) {
	//	for (int i = 0; i < iter->second.size(); i++) {
	//		count++;
	//		if (count % 100 == 0)
	//			printf("%d\n", count);
	//		string filename = "";
	//		filename.append(dir);
	//		filename.append(iter->second[i]);
	//		filename.append(".wav");
	//		MfccExtractor::getMfccFromWav(filename.c_str(), &iFrameNum, &mInput);
	//		string result = recognizer->recognize(mInput, iFrameNum, nodeHolder);
	//		//cout << result << " " << iter->first << endl;

	//		if (result == iter->first)
	//			correct++;
	//	}
	//}

	//printf("accuracy: %d / %d = %f\n", correct, count, (float)correct / count);
	//------------

	PaStreamParameters  inputParameters,
                        outputParameters;
    PaError             err = paNoError;
    paTestData          data;
    int                 i = 0;
    int                 totalFrames = 0;
    int                 numSamples = 0;

	totalFrames = NUM_SECONDS * SAMPLE_RATE;
    numSamples = totalFrames * NUM_CHANNELS;
	data.recordedSamples = (SAMPLE *)malloc(numSamples * sizeof(SAMPLE));

    err = Pa_Initialize();
    if( err != paNoError ) exit(1);

    inputParameters.device = Pa_GetDefaultInputDevice(); /* default input device */
    if (inputParameters.device == paNoDevice) {
        fprintf(stderr,"Error: No default input device.\n");
        exit(1);
    }
    inputParameters.channelCount = 1;                    /* stereo input */
    inputParameters.sampleFormat = PA_SAMPLE_TYPE;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = NULL;

	while (true) {
		data.maxFrameIndex = NUM_SECONDS * SAMPLE_RATE; /* Record for a few seconds. */
		data.frameIndex = 0;
		data.existSpeech = false;
		data.speechCount = 0;
		data.silenceCount = 0;
		for( i=0; i<numSamples; i++ ) 
			data.recordedSamples[i] = 0;

		printf("please hit to speak.\n");
		bool exit = false;
		char c = getchar();
		if (c == 27)
			break;
		PaStream* stream;
		err = Pa_OpenStream(
			&stream,
			&inputParameters,
			NULL,
			SAMPLE_RATE,
			FRAMES_PER_BUFFER,
			paClipOff,
			recordCallback,
			&data );
		if( err != paNoError )
			break;

		err = Pa_StartStream( stream );
		if( err != paNoError )
			break;
		printf("Now recording..."); 
		fflush(stdout);

		while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
		{
			printf("==\n");
			Pa_Sleep(250);
		}
		if( err < 0 )
			break;
		err = Pa_CloseStream( stream );
		if( err != paNoError )
			break;

		//====
		char *filename = "test.wav";
		int _start = data.SpeechStart - SILENCE_LENGTH * FRAMES_PER_BUFFER;
		_start = (_start < 0 ? 0 : _start);
		WriteWave(filename, &data.recordedSamples[_start], data.frameIndex - _start, SAMPLE_RATE);
		//====

		// begin test
		int start = data.SpeechStart - SILENCE_LENGTH * FRAMES_PER_BUFFER;
		start = (start < 0 ? 0 : start);
		int iFrameNum = 0;
		double **mInput = NULL;
		//MfccExtractor::getMfccFromWav("hwdata/test/FAK_86Z1162A.wav", &iFrameNum, &mInput);
		MfccExtractor::getMfcc(&data.recordedSamples[start], data.frameIndex - start, SAMPLE_RATE, &iFrameNum, &mInput);
		string result = recognizer->recognize(mInput, iFrameNum, nodeHolder);
		cout << "result: " << result << endl;
	}

    Pa_Terminate();
    if( data.recordedSamples )       /* Sure it is NULL or valid. */
        free( data.recordedSamples );
    if( err != paNoError )
    {
        fprintf( stderr, "An error occured while using the portaudio stream\n" );
        fprintf( stderr, "Error number: %d\n", err );
        fprintf( stderr, "Error message: %s\n", Pa_GetErrorText( err ) );
        err = 1;          /* Always return 0 or 1, but no other return codes. */
    }
}

