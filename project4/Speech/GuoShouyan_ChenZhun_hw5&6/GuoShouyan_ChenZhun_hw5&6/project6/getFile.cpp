#include <string>
#include <io.h>
#include <vector>
#include <iostream>
#include "getFile.h"

using namespace std;

/************************************************************************/
/*  获取文件夹下所有文件名
输入：		
path	:	文件夹路径
exd		:   所要获取的文件名后缀，如jpg、png等；如果希望获取所有
文件名, exd = ""
输出：
files	:	获取的文件名列表

*/
/************************************************************************/

void getFiles( string path, string exd, vector<string>& files )
{
	//文件句柄
	long   hFile   =   0;
	//文件信息
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(),&fileinfo)) !=  -1)
	{
		do
		{
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
			if((fileinfo.attrib &  _A_SUBDIR))
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					getFiles( pathName.assign(path).append("\\").append(fileinfo.name), exd, files );
			}
			else
			{
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		}while(_findnext(hFile, &fileinfo)  == 0);
		_findclose(hFile);
	}
}

