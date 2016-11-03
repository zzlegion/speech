#ifndef _READWAV_H_
#define _READWAV_H_

#include <stdio.h>
#include "portaudio.h"

struct WavFileHead
{
	//Resource Interchange File Flag (0-3) "RIFF"，
	char RIFF[4];

	//File Length ( not include 8 bytes from the beginning ) (4-7) 文件长度
	int FileLength;

	//WAVE File Flag (8-15) "WAVEfmt ", 字符 波形格式标志“fmt ”
	char WAVEfmt_[8];

	//Transitory Byte ( normally it is 10H 00H 00H 00H ) (16-19)
	unsigned int noUse;

	//Format Category ( normally it is 1 means PCM-u Law ) (20-21) 整形 格式类别（值＝1表示编码方式为PCMμ律编码）
	short FormatCategory;

	//NChannels (22-23) 整形 声道数（单声道＝1，双声音＝2）
	short NChannels;

	//Sample Rate (24-27) 长整形   采样频率（每秒样本数，表示每个通道的播放速度）
	int SampleRate; 

	//l=NChannels*SampleRate*NBitsPersample/8 (28-31) 1C 长整形   数据传送速率（每秒字节＝采样频率×每个样本字节数）
	int SampleBytes;

	//i=NChannels*NBitsPersample/8 (32-33) 整形     每个样本字节数（又称基准块＝每个样本位数×声道数÷8）
	short BytesPerSample;

	//NBitsPersample (34-35)  整形     每个样本位数（又称量化位数）
	short NBitsPersample;

	//Data Flag (36-39) "data"  整形 附加信息（可选，通过块长度来判断有无）
	char data[4];

	//Raw Data File Length (40-43) 原始数据文件长度
	int RawDataFileLength;

};

// original functions
bool	WaveRewind(FILE *wav_file, WavFileHead *wavFileHead);
short	*ReadWave(const char *wavFile, int *numSamples, int *sampleRate);
void	WriteWave(const char *wavFile, short *waveData, int numSamples, int sampleRate);
void	FillWaveHeader(void *buffer, int raw_wave_len, int sampleRate);

// additive functions
void    GetWavHeader(const char *wavFile,short *Bits,int *Rate,short *Format,int *Length,short *Channels);
short   *ReadWavFile(const char *wavFile, int *numSamples, int *sampleRate);
void    readwav_t(const char *wavFile, short *waveData, long times, int *numSamples, int *sampleRate);
void    GetWavTime(const char *wavFile, double *duration);
void    ReadWav(const char *wavFile, short *waveData, int *numSamples, int *sampleRate);

#endif //_READWAV_H_
