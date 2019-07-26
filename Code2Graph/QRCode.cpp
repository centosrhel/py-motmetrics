//
//  QRCode.cpp
//  RS_CRC_Code
//
//  Created by Hu Yafei on 23/03/2018.
//

#include "QRCode.h"
#include "interface.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef DEBUG
#include <opencv2/highgui.hpp>
#include <iostream>
#endif

using namespace cv;
using namespace std;

namespace Ali_QRCode{
#define DEFAULT_CODE_LEN 5
#define DEFALUT_INFO_LEN 2
//const int NORMALIZED_WIDTH = 430;
const int patternRadius = 105;
const int border_width = NORMALIZED_WIDTH / 2 - patternRadius;

QRCode::QRCode(const char* strDirectory, const unsigned num_pictures) : m_strDirectory(string(strDirectory)), m_num_pictures(num_pictures){
    //m_contours.clear();
    //m_contours2.clear();
    m_color = Scalar(0xff, 0x82, 0x00);
    //m_color = Scalar(0x89, 0x20, 0x00);
}

int QRCode::generateQRCode(unsigned long ID, unsigned char* outputImgData)
{
    int img_ordinal = ID % m_num_pictures;
    string filename_full = m_strDirectory + string("/") + to_string(img_ordinal) + string(".jpg");
    Mat img = imread(filename_full, 0);
    if(img.empty()) return -1;
    resize(img, img, Size(2*patternRadius, 2*patternRadius));
    threshold(img, img, 128, 255, THRESH_OTSU);
    Mat imgBGR;
    cvtColor(img, imgBGR, CV_GRAY2BGR);

    int nRows = imgBGR.rows, nCols = imgBGR.cols;
    float centerX = nRows/2-0.5, centerY = nCols/2-0.5, thres_radius = centerX*centerY;
    Vec3b* p;
    for (int i = 0; i < nRows; ++i)
    {
        p = imgBGR.ptr<Vec3b>(i);
        for (int j = 0; j < nCols; ++j)
        {
            if((i-centerX)*(i-centerX) + (j - centerY)*(j-centerY) > thres_radius)
                p[j][0] = p[j][1] = p[j][2] = 255;
            else if (p[j][0] == 255){
//                p[j][0] = 137;
//                p[j][1] = 32;
//                p[j][2] = 0;
            }
            else{
                p[j][0] = 0xff;
                p[j][1] = 0x82;
                p[j][2] = 0;
//                p[j][0] = p[j][1] = p[j][2] = 255;
            }
        }
    }

    Mat enlargedImg;
    copyMakeBorder(imgBGR, enlargedImg, border_width, border_width, border_width, border_width, BORDER_CONSTANT, Scalar_<uchar>(255,255,255));

    uchar information[DEFALUT_INFO_LEN];
    information[0] = (ID&0x00FF);
    information[1] = ((ID&0xFF00)>>8);

    uchar encodeInfo[DEFAULT_CODE_LEN];
    int outLen = 0;
    encodeCRCAndRS(information, DEFALUT_INFO_LEN, encodeInfo, outLen);
    if(outLen != DEFAULT_CODE_LEN) return -2;

    code2Image(enlargedImg, encodeInfo, DEFAULT_CODE_LEN, m_color);

    //rectangle(enlargedImg, Point(75,75), Point(105,105), Scalar(255,255,255), CV_FILLED, LINE_AA);
    circle(enlargedImg, Point(95,95), 20, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(95,95), 15, m_color, -1, LINE_AA);
    circle(enlargedImg, Point(95,95), 10, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(95,95), 3, m_color, -1, LINE_AA);

    //rectangle(enlargedImg, Point(314,75), Point(354,105), Scalar(255,255,255), CV_FILLED, LINE_AA);
    circle(enlargedImg, Point(334,95), 20, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(334,95), 15, m_color, -1, LINE_AA);
    circle(enlargedImg, Point(334,95), 10, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(334,95), 3, m_color, -1, LINE_AA);

    //rectangle(enlargedImg, Point(75,314), Point(105,354), Scalar(255,255,255), CV_FILLED, LINE_AA);
    circle(enlargedImg, Point(95,334), 20, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(95,334), 15, m_color, -1, LINE_AA);
    circle(enlargedImg, Point(95,334), 10, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(95,334), 3, m_color, -1, LINE_AA);

    //rectangle(enlargedImg, Point(314,314), Point(354,354), Scalar(255,255,255), CV_FILLED, LINE_AA);
    circle(enlargedImg, Point(334,334), 20, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(334,334), 15, m_color, -1, LINE_AA);
    circle(enlargedImg, Point(334,334), 10, Scalar_<uchar>(255,255,255), -1, LINE_AA);
    circle(enlargedImg, Point(334,334), 3, m_color, -1, LINE_AA);

    //unsigned char outputImgData[sizeof(char)*NORMALIZED_SIZE];
    memcpy(outputImgData, enlargedImg.data, enlargedImg.elemSize()*enlargedImg.total());

    return 0;
}

int QRCode::recognizeQRCode(const Ali_Image& ali_image, unsigned long& ID)
{
    Mat img(ali_image.height, ali_image.width, CV_8UC3, ali_image.data);
#ifdef DEBUG
    ////////////////
    imshow("img", img);
    //////////////
#endif
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 75, 150, 3);
#ifdef DEBUG
    //////////////////
    imshow("edges", edges);
    waitKey();
    ///////////////////
#endif
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //int mark  = 0, A, B, C, D;
    int mark = 0;
    m_contours.clear();

    for (int i = 0; i < contours.size(); ++i) {
        int k = i, c = 0;
        while (hierarchy[k][2] != -1) {
            k = hierarchy[k][2];
            ++c;
        }
        if (c == 5) {
//            if (mark == 0) A = i;
//            else if(mark == 1) B = i;
//            else if(mark == 2) C = i;
//            else if(mark == 3) D = i;
            int kk = i;
            for(int j = 0; j < 4; ++j)
                kk = hierarchy[kk][2];
            double first_area = contourArea(contours[i]);
            double fifth_area = contourArea(contours[kk]);
            double first_vs_fifth = first_area/(fifth_area + 1e-8);
#ifdef DEBUG
            cout << first_area <<' ' << fifth_area << ' ' <<first_vs_fifth << endl;
#endif
            if(first_vs_fifth >= 15 && first_vs_fifth <= 50)
            {
                mark++;
                m_contours.push_back(contours[i]);
            }
        }
    }
#ifdef DEBUG
    cout<<mark<<endl;
#endif

    if(mark == 4 || mark == 5)
    {
        contours.clear();
        if(mark == 5)
        {
            unsigned max_perimeter = 0, contour_id;
            for(int i = 0; i < 5; ++i)
            {
                //cout << m_contours[i].size() << endl;
                if(m_contours[i].size() > max_perimeter)
                {
                    max_perimeter = m_contours[i].size();
                    contour_id = i;
                }
            }
            //cout << max_perimeter << ' ' << contour_id << endl;
            //m_contours2.resize(4);
            //cout << m_contours2.size() << endl;
            for(int j = 0; j < 5; ++j)
            {
                //cout << m_contours[j].size() << endl;
                if(j == contour_id) continue;
                contours.push_back(m_contours[j]);
                //cout << m_contours2[0].size() <<' ' << m_contours[j].size() << endl;
            }
            //cout << m_contours2.size() << endl;
        }
        else
        {
            for(int j = 0; j < mark; ++j)
                contours.push_back(m_contours[j]);
        }
        vector<Moments> mu;
        vector<Point2f> mc;
        mu.resize(4);
        mc.resize(4);
        for(int i = 0; i < 4; ++i)
        {
            //cout << m_contours2[i].size() << endl;
            mu[i] = moments(contours[i], false);
            mc[i] = Point2f(mu[i].m10/mu[i].m00, mu[i].m01/mu[i].m00);
        }
//        mu[0] = moments(contours[A], false);
//        mc[0] = Point2f(mu[0].m10/mu[0].m00, mu[0].m01/mu[0].m00);
//        mu[1] = moments(contours[B], false);
//        mc[1] = Point2f(mu[1].m10/mu[1].m00, mu[1].m01/mu[1].m00);
//        mu[2] = moments(contours[C], false);
//        mc[2] = Point2f(mu[2].m10/mu[2].m00, mu[2].m01/mu[2].m00);
//        mu[3] = moments(contours[D], false);
//        mc[3] = Point2f(mu[3].m10/mu[3].m00, mu[3].m01/mu[3].m00);

        vector<Point2f> src, dst;
        src.push_back(mc[0]);
        src.push_back(mc[1]);
        src.push_back(mc[2]);
        src.push_back(mc[3]);
        vector<Point2f> src_final;
        src_final.resize(4);
        Point2f centerPoint;
        for(int i = 0; i < 4; ++i)
            centerPoint += src[i];
        centerPoint /= 4;
        for(int i = 0; i < 4; ++i)
        {
            if(src[i].y < centerPoint.y)
            {
                if (src[i].x < centerPoint.x)
                    src_final[0] = src[i];
                else src_final[1] = src[i];
            }
            else if(src[i].x > centerPoint.x) src_final[2] = src[i];
            else src_final[3] = src[i];
        }

        Mat warp_matrix;
        dst.push_back(Point2f(95,95));
        dst.push_back(Point2f(334,95));
        dst.push_back(Point2f(334, 334));
        dst.push_back(Point2f(95, 334));

        warp_matrix = getPerspectiveTransform(src_final, dst);
        Mat qr_raw = Mat::zeros(NORMALIZED_HEIGHT, NORMALIZED_WIDTH, CV_8UC3);
        Mat qr_gray = Mat::zeros(NORMALIZED_HEIGHT, NORMALIZED_WIDTH, CV_8U);
        Mat qr_thres = Mat::zeros(NORMALIZED_HEIGHT, NORMALIZED_WIDTH, CV_8U);
        warpPerspective(img, qr_raw, warp_matrix, Size(NORMALIZED_WIDTH, NORMALIZED_HEIGHT));
        cvtColor(qr_raw, qr_gray, CV_BGR2GRAY);
#ifdef DEBUG
        /////////////////
        imshow("qr_raw", qr_raw);
        imshow("qr_gray", qr_gray);
        ////////////////
#endif
        threshold(qr_gray, qr_thres, 128, 255, THRESH_OTSU);
        medianBlur(qr_thres, qr_thres, 3);
#ifdef DEBUG
        ////////////////
        imshow("qr_binary", qr_thres);
        waitKey();
        imwrite("qr_binary.png", qr_thres);
        ////////////////
#endif
        uchar imageCodeResult[DEFAULT_CODE_LEN];
        if(!image2Code(qr_thres, DEFAULT_CODE_LEN, imageCodeResult)) return -2;

        uchar decodeInfo[DEFALUT_INFO_LEN];
        if(!decodeCRCAndRS(imageCodeResult, DEFAULT_CODE_LEN, DEFALUT_INFO_LEN, decodeInfo)) return -3;
        ID = decodeInfo[0] + (decodeInfo[1] << 8);
        return 0;
    }
    else return -1;
}
} // end namespace Ali_QRCode
