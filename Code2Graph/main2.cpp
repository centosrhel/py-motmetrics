#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "interface.h"
using namespace std;


int main()
{
    uchar code[]={'a','b','c','d','e'};
//    uchar code[]={0,0,0,0,0};
//    uchar code[]={255,255,255,255,255};
//    cv::Mat codeImage(IMAGE_SIZE,IMAGE_SIZE,CV_8UC3);
//    codeImage.setTo(cv::Scalar(255,255,255));

    cv::Mat codeImage=cv::imread("/home/sixd-ailabs/Desktop/input2.png",cv::IMREAD_COLOR);
//    test(codeImage);
    codeImage.setTo(cv::Scalar(255,255,255));
    code2Image(codeImage,code,5,cv::Scalar(0xFF,0x82,0x00));

    cv::imwrite("/home/sixd-ailabs/Desktop/code.png",codeImage);

//    cv::Mat tmpImage;
//    cv::flip(codeImage,tmpImage,0);
//    cv::flip(tmpImage,codeImage,1);
//    cv::rotate(codeImage,codeImage,cv::ROTATE_90_CLOCKWISE);
//    cv::rotate(codeImage,codeImage,cv::ROTATE_90_CLOCKWISE);
//    cv::rotate(codeImage,codeImage,cv::ROTATE_90_CLOCKWISE);
    while (true) {

        cv::imshow("CodeImage",codeImage);
        if(cv::waitKey(10)==27) break;
    }
    cv::Mat grayImage=cv::imread("/home/sixd-ailabs/Desktop/lALPBbCc1Yv8qlbNAa7NAa4_430_430.png",cv::IMREAD_GRAYSCALE);
//    cv::cvtColor(codeImage,grayImage,CV_BGR2GRAY);
    unsigned char result[5];
    image2Code(grayImage,5,result);
    for(int i=0;i<5;i++)
    {
        std::cout<<result[i]<<" ";
//        assert(code[i]==result[i]);
    }
    unsigned char decodeInfo[2];
    bool crcAns = decodeCRCAndRS(result, 5, 2, decodeInfo);

        assert(crcAns);
    std::cout<<std::endl;

    return 0;
}
