
#include "IQRCode.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace Ali_QRCode;
using namespace std;


float calculateAccuracy(IQRCode* pIQRCode,const std::string &path,int &total,int &recognized)
{
    struct dirent *ptr;
    DIR *dir;
    dir=opendir(path.c_str());
    vector<string> files;
    while((ptr=readdir(dir))!=NULL)
    {

        //跳过'.'和'..'两个目录
        if(ptr->d_name[0] == '.')
            continue;
        //cout << ptr->d_name << endl;
        std::string name=ptr->d_name;
        if(name.find(".jpg")>=0)
            files.push_back(name);
    }
    closedir(dir);

    recognized=0;
    total=files.size();
    if(total==0 ) return 100;

    for(int i=0;i<total;i++)
    {
        cv::Mat img=cv::imread(path+"/"+files[i]);
        Ali_Image ali_image;
        ali_image.data = img.data;
        ali_image.height = img.rows;
        ali_image.width = img.cols;
        ali_image.nChannels = img.channels();
        unsigned long ID = 0;
        int res = pIQRCode->recognizeQRCode(ali_image, ID);
        if(res == 0)
        {
            recognized++;
//            cout<< files[i]<< " " << ID << endl;
        }
        else if(res == -1) cout<< files[i] << " Error Positioning!" << endl;
        else if(res == -2) cout<< files[i] << " Error image2Code" << endl;
        else if(res == -3) cout<< files[i] << " Error decodeCRCAndRS" << endl;
    }

    return 100.0*recognized/total;
}


int main(int argc, const char* argv[])
{
    std::shared_ptr<IQRCode> pIQRCode = makeIQRCode("/home/sixd-ailabs/Develop/QRCode/ChladniImagesRename", 279);
    std::vector<string> testDataSet;
    testDataSet.push_back("/home/sixd-ailabs/Develop/QRCode/testData/S1");

    int sumTotal=0;
    int sumRecognized=0;
    for(string path:testDataSet)
    {
        int total,recognized;
        float accuracy=calculateAccuracy(pIQRCode.get(),path,total,recognized);
        std::cout<<path<<"------"<<accuracy<<"%"<<std::endl;
        sumTotal+=total;
        sumRecognized+=recognized;
    }
    if(sumTotal>0)
    {
        float result=100.0*sumRecognized/sumTotal;
        std::cout<<"Accuracy:"<<result<<"%"<<std::endl;
    }
    else
    {
        std::cout<<"No test data."<<std::endl;
    }

    return 0;
}
