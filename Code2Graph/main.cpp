#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace std;

static const float CODE_RADIAN[]={1,5,10,15};
static const float CODE_HIT_INTERVAL[4][2] ={{5,13},{15,24},{28,38},{44,54}};
static const int DRAW_SEGMENT_WIDTH=3;
static const int MAX_RADIUS=190;
static const int CIRCLE_INTERVAL=20;
static const float DRAW_STEP=DRAW_SEGMENT_WIDTH*1.0/MAX_RADIUS;
static const int IMAGE_SIZE=430;
static const uchar MASK=0x96;//1001 0110

void drawOneCode(cv::Mat &codeImage, float radius,float radian,int code,const cv::Scalar &color)
{
    int centerX=codeImage.cols/2;
    int centerY=codeImage.rows/2;

    const double drawStepResolution=10.0;
    int stepCnt=CODE_RADIAN[code]==1?1:(CODE_RADIAN[code]-1)*drawStepResolution;
    for(int i=0;i<stepCnt;i++)
    {
        int x=radius*cos(radian)+centerX;
        int y=-radius*sin(radian)+centerY;
        if(stepCnt==1)
            cv::circle(codeImage,cv::Point2f(x,y),DRAW_SEGMENT_WIDTH+1,color,-1,cv::LINE_AA);
        else
            cv::circle(codeImage,cv::Point2f(x,y),DRAW_SEGMENT_WIDTH,color,-1,cv::LINE_AA);
        radian+=(DRAW_STEP/drawStepResolution);
    }
}

float getCodeAngle(int code)
{
    return DRAW_STEP*(CODE_RADIAN[code]+1)/M_PI*180;
}

void test(cv::Mat &codeImage)
{
    for(int j=0;j<2;j++)
    for(int i=0;i<3;i++)
    {
        double rad=i*20.0/180*M_PI;
        drawOneCode(codeImage,200-j*20,0+rad,0,cv::Scalar(225,105,65));
        drawOneCode(codeImage,200-j*20,M_PI_2+rad,1,cv::Scalar(225,105,65));
        drawOneCode(codeImage,200-j*20,M_PI+rad,2,cv::Scalar(225,105,65));
        drawOneCode(codeImage,200-j*20,-M_PI_2+rad,3,cv::Scalar(225,105,65));
    }
}

std::vector<std::vector<int>> splitVector(const std::vector<int> codeVector,int num)
{
    int size=codeVector.size();
    int newSize=(size+1)/num;
    std::vector<std::vector<int>> result;
    result.reserve(num);
    for(int i=0;i<num-1;i++)
    {
        std::vector<int> circle;
        for(int j=0;j<newSize;j++)
        {
            circle.push_back(codeVector[i*newSize+j]);
        }
        result.push_back(circle);
    }
    std::vector<int> circle;
    for(int j=newSize*(num-1);j<size;j++)
    {
        circle.push_back(codeVector[j]);
    }
    result.push_back(circle);
    return result;
}

inline float angleToRadian(float angle)
{
    return angle/180.0*M_PI;
}

float sumOfCircleCodeAngle(const std::vector<int> &circleCode)
{
    float totalAngle=0;
    for(int c:circleCode)
    {
        totalAngle+=getCodeAngle(c);
    }
    return totalAngle;
}

bool drawCircleCode(cv::Mat &codeImage, const std::vector<int> &circleCode,int radius,int startRadian,const cv::Scalar &color)
{
    float totalAngle=sumOfCircleCodeAngle(circleCode);
    if(totalAngle>360)
        return false;

    float spaceIntervalRadian=angleToRadian((360-totalAngle)/(circleCode.size()+0.5));
    float radian=startRadian;
    for(int c:circleCode)
    {
        radian+=DRAW_STEP;
        drawOneCode(codeImage,radius,radian,c,color);
        radian-=DRAW_STEP;
        radian+=angleToRadian(getCodeAngle(c));
        radian+=spaceIntervalRadian;
//        cv::imshow("test2",codeImage);
//        if(cv::waitKey(1000)==27) break;
    }
    return true;
}

bool code2Image(cv::Mat &codeImage,const uchar code[],int length,const cv::Scalar &color)
{
    if(length>5)
        return false;

    std::vector<int> codeVector;
    codeVector.reserve(length*4);
//    test(codeImage);

    for(int i=0;i<length;i++)
    {
        uchar v=(code[i]^MASK);
       for(int j=0;j<4;j++)
       {
           int code=(v&0x03);
           codeVector.push_back(code);
           v=(v>>2);
       }
    }
    int radius=MAX_RADIUS;
    drawCircleCode(codeImage,codeVector,radius,0,color);
//    std::vector<std::vector<int>> circleCodeVectors=splitVector(codeVector,2);
//    for(int i=0;i<circleCodeVectors.size();i++)
//    {
//        drawCircleCode(codeImage,circleCodeVectors[i],radius,0,color);
//        radius-=CIRCLE_INTERVAL;
//    }

    return true;
}
inline bool hitTest(const cv::Mat& codeImageBinary,int row,int col)
{
    if(codeImageBinary.at<uchar>(row,col)!=255)
        return true;
    else
        return false;
}

int parseHitCnt(int hitCnt)
{
    for(int i=0;i<4;i++)
    {
        if(hitCnt>=CODE_HIT_INTERVAL[i][0]&&hitCnt<=CODE_HIT_INTERVAL[i][1])
            return i;
    }
    return -1;
}

bool parseCodeVector(const std::vector<int> &codeVector,uchar code[],int &length)
{
    if(codeVector.size()%4!=0)
        return false;
    length=codeVector.size()/4;
    for(int i=0;i<length;i++)
    {
        int v=0;
        for(int j=0;j<4;j++)
        {
            int index=i*4+j;
            v+=(codeVector[index]<<(2*j));
        }
        code[i]=v;
        code[i]=(code[i]^MASK);
    }
    return true;
}

bool image2Code(const cv::Mat &codeImageBinary,uchar code[],int &length)
{
    int radius=MAX_RADIUS;
    int centerX=codeImageBinary.cols/2;
    int centerY=codeImageBinary.rows/2;
    float step=1.0/MAX_RADIUS;
    int hitCnt=0;
    int spaceCnt=0;
    std::vector<int> codeVector;
    std::vector<int> spaceVector;
    for(float radian=0;radian<M_PI*2-step;radian+=step)
    {
        int x=radius*cos(radian)+centerX;
        int y=-radius*sin(radian)+centerY;

        if(hitTest(codeImageBinary,y,x))
        {
            hitCnt++;
            if(spaceCnt!=0)
            {
                spaceVector.push_back(spaceCnt);
            }
            spaceCnt=0;
        }
        else
        {
            if(hitCnt!=0)
            {
                int value=parseHitCnt(hitCnt);
                if(value!=-1)
                {
                    codeVector.push_back(value);
                }
            }
            hitCnt=0;
            spaceCnt++;
        }
    }
    if(spaceCnt!=0)
        spaceVector.push_back(spaceCnt);

    int lastSpaceCnt=spaceVector.back();
    for(int i=0;i<spaceVector.size()-1;i++)
    {
        if(spaceVector[i]*1.2>lastSpaceCnt)
            return false;
    }

    return parseCodeVector(codeVector,code,length);
}

int main()
{
//    uchar code[]={'a','b','c','d','e'};
    uchar code[]={255,255,255,255,255};
//    cv::Mat codeImage(IMAGE_SIZE,IMAGE_SIZE,CV_8UC3);
//    codeImage.setTo(cv::Scalar(255,255,255));

    cv::Mat codeImage=cv::imread("/home/sixd-ailabs/Desktop/input2.png",cv::IMREAD_COLOR);

    if(code2Image(codeImage,code,5,cv::Scalar(225,105,65)))
    {
        cv::imwrite("/home/sixd-ailabs/Desktop/code.png",codeImage);
        cv::Mat grayImage;
        cv::cvtColor(codeImage,grayImage,CV_BGR2GRAY);
    //    while (true) {

    //        cv::imshow("Gray",grayImage);
    //        if(cv::waitKey(10)==27) break;
    //    }
        uchar result[5];
        int length;
        if(image2Code(grayImage,result,length))
        {
            assert(length==5);
            for(int i=0;i<length;i++)
            {
                std::cout<<result[i]<<" ";
                assert(code[i]==result[i]);
            }
            std::cout<<std::endl;
        }

    }

//    while (true) {

//        cv::imshow("CodeImage",codeImage);
//        if(cv::waitKey(10)==27) break;
//    }

    return 0;
}
