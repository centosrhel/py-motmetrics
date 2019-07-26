#include <iostream>
#include "interface.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;
#define DEFAULT_CODE_LEN 5
#define DEFALUT_INFO_LEN 2
int main()
{

    ushort id=65535;

    uchar information[DEFALUT_INFO_LEN];
    information[0]=(id&0x00FF);
    information[1]=((id&0xFF00)>>8);

    uchar encodeInfo[DEFAULT_CODE_LEN];
    int outLen = 0;
    encodeCRCAndRS(information, 2, encodeInfo, outLen);

    assert(outLen==DEFAULT_CODE_LEN);

    cv::Mat codeImage=cv::imread("/home/sixd-ailabs/Desktop/input2.png",cv::IMREAD_COLOR);
    codeImage.setTo(cv::Scalar(255,255,255));
    code2Image(codeImage,encodeInfo,DEFAULT_CODE_LEN,cv::Scalar(225,105,65));

//    cv::RotatedRect rect(cv::Point2f(codeImage.cols/2,codeImage.rows/2),cv::Size(170,170),45);
//    cv::ellipse(codeImage,cv::Point2f(codeImage.cols/2,codeImage.rows/2),cv::Size(170,170),0,0,-10,cv::Scalar(255,0,0),6,cv::LINE_AA);
//    cv::line(codeImage,cv::Point(0,0),cv::Point(codeImage.cols,codeImage.rows),cv::Scalar(255,0,0),10);
//    cv::line(codeImage,cv::Point(codeImage.cols,0),cv::Point(0,codeImage.rows),cv::Scalar(255,0,0),10);


    cv::Mat grayImage;
    cv::cvtColor(codeImage,grayImage,CV_BGR2GRAY);
    uchar imageCoderResult[DEFAULT_CODE_LEN];
    bool bIC=image2Code(grayImage,5,imageCoderResult);
    assert(bIC);

    uchar decodeInfo[DEFALUT_INFO_LEN];
    bool crcAns = decodeCRCAndRS(imageCoderResult, DEFAULT_CODE_LEN, DEFALUT_INFO_LEN, decodeInfo);

    assert(crcAns);
    ushort decodeValue=decodeInfo[0]+(decodeInfo[1]<<8);
    std::cout<<"id="<<id<<std::endl;
    std::cout<<"decodeValue="<<decodeValue<<std::endl;
    assert(decodeValue==id);

    cv::imshow("Des",codeImage);
    cv::imwrite("example.png", codeImage);
    while (true) {
        if( cv::waitKey(300)==27) break;
    }

    return 0;
}
