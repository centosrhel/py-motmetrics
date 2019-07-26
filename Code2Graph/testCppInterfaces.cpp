//
//  testCppInterfaces.cpp
//  RS_CRC_Code
//
//  Created by Hu Yafei on 26/03/2018.
//

#include "IQRCode.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace Ali_QRCode;
using namespace std;

int main(int argc, const char* argv[])
{
//    IQRCode* pIQRCode = createIQRCode("/Users/hu/Documents/FaceX/ChladniPatterns_279_renamed", 279);
    shared_ptr<IQRCode> pIQRCode = makeIQRCode("/media/sixd-ailabs/data_disk/QrCode/ChladniImagesRename", 279);
    stringstream ss;
    ss << argv[1];
    unsigned long ID;
    ss >> ID;

    Mat img(NORMALIZED_HEIGHT, NORMALIZED_WIDTH, CV_8UC3);
    int error_num = 0;
    for(unsigned int i = 0; i < 1; ++i)
    {
        //Mat img(NORMALIZED_HEIGHT, NORMALIZED_WIDTH, CV_8UC3);
        int res = pIQRCode->generateQRCode(i, img.data);
//        int res = pIQRCode->generateQRCode(ID, img.data);

        //ss.str("");
        //ss << i;
        if(res == 0) imwrite(to_string(i)+".png", img);
//        if(res == 0) imwrite(to_string(ID)+".png", img);

        img = imread("/home/sixd-ailabs/Downloads/qrcode_bad_case/bad_case_3.png",1);
        cv::imshow("img",img);
        cv::waitKey(0);

        Ali_Image ali_image;
        ali_image.data = img.data;
        ali_image.height = img.rows;
        ali_image.width = img.cols;
        ali_image.nChannels = img.channels();
        unsigned long ID2 = 0;
        res = pIQRCode->recognizeQRCode(ali_image, ID2);

        if(res == 0 /*&& ID2 != i*/)
        {
//            cout << boolalpha << (ID2 == i) << endl;
            cout << boolalpha << false << endl;
            cout << ID2 <<endl;
            //error_num++;
        }
        else if(res == -1)
        {
            cout << i << " Error Positioning!" << endl;
            //system("pause");
            error_num++;
        }
        else if(res == -2)
        {
            cout << i << " Error image2Code" << endl;
            //system("pause");
            error_num++;
        }
        else if(res == -3)
        {
            cout << i << " Error decodeCRCAndRS" << endl;
            //system("pause");
            error_num++;
        }
    }
    cout << "error_num: " << error_num << endl;

//    releaseIQRCode(&pIQRCode);
    return 0;
}
