
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "interface.h"
using namespace std;
static const int EMPTY=0;
static const int POINT=1;
static const int SEGMENT=2;
static const int TOTAL_SECTION=48;


static const int DRAW_SEGMENT_WIDTH=3;
static const int MAX_RADIUS=208;
static const int MIN_RADIUS=123;
static const int CIRCLE_INTERVAL=20;
static const float DRAW_STEP=DRAW_SEGMENT_WIDTH*1.0/MAX_RADIUS;
static const uchar MASK=0x96;//1001 0110
static const int CODE_HIT_INTERVAL_MAX[2][2] ={{4,13},{14,30}};
static const int CODE_HIT_INTERVAL_MIN[2][2] ={{5,20},{21,30}};

static inline float radianToAngle(float radian)
{
    return radian/M_PI*180;
}

void drawSections(cv::Mat &codeImage, float radius,int totalSections,int begin,int end,int code,const cv::Scalar &color,float offset)
{
    const int centerX=codeImage.cols/2;
    const int centerY=codeImage.rows/2;

    float sectionLen=M_PI*2/totalSections;

//    if(radius==MIN_RADIUS)
//    {
//        int xt=radius*cos(begin*sectionLen+offset)+centerX;
//        int yt=-radius*sin(begin*sectionLen+offset)+centerY;
//        cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(xt,yt),color);
//    }


    if(code==EMPTY)
        return;
    float radian=begin*sectionLen+DRAW_STEP+offset;
    if(code==POINT)
        radian=(begin+0.5)*sectionLen+offset;
//    const double drawStepResolution=10.0;

    if(code==POINT)
    {
        int stepCnt=end-begin;

        for(int i=0;i<stepCnt;i++)
        {
            int x=radius*cos(radian)+centerX;
            int y=-radius*sin(radian)+centerY;

            cv::circle(codeImage,cv::Point2f(x,y),DRAW_SEGMENT_WIDTH,color,-1,cv::LINE_AA);

            radian+=sectionLen;

        }
    }
    else
        cv::ellipse(codeImage,cv::Point2f(centerX,centerY),cv::Size(radius,radius),0,
                    -radianToAngle(begin*sectionLen+DRAW_STEP+offset),
                    -radianToAngle(end*sectionLen-DRAW_STEP+offset),
                    color,DRAW_SEGMENT_WIDTH,cv::LINE_AA);


//    int stepCnt=code==1?end-begin:((sectionLen*(end-begin))/DRAW_STEP-2)*drawStepResolution;

//    for(int i=0;i<stepCnt;i++)
//    {
//        int x=radius*cos(radian)+centerX;
//        int y=-radius*sin(radian)+centerY;
//        if(code==POINT)
//            cv::circle(codeImage,cv::Point2f(x,y),DRAW_SEGMENT_WIDTH+1,color,-1,cv::LINE_AA);
//        else
//            cv::circle(codeImage,cv::Point2f(x,y),DRAW_SEGMENT_WIDTH,color,-1,cv::LINE_AA);
//        if(code==POINT)
//            radian+=sectionLen;
//        else
//            radian+=(DRAW_STEP/drawStepResolution);
//    }
}

void test(cv::Mat &codeImage)
{
    for(int j=0;j<2;j++)
    for(int i=0;i<3;i++)
    {
        double rad=i*20.0/180*M_PI;
        drawSections(codeImage,200-j*20,48,0,1,0,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,1,2,1,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,2,6,2,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,6,7,0,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,7,8,2,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,8,9,0,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,9,12,1,cv::Scalar(225,105,65),0);
        drawSections(codeImage,200-j*20,48,12,13,0,cv::Scalar(225,105,65),0);

    }
}

std::vector<int> generateSectionCode(const uchar code[],int length)
{
    std::vector<int> result;
    result.reserve(length*8);
    for(int k=0;k<length;k++)
    {
        uchar c=(code[k]^MASK);
        for(int i=0;i<8;i++)
        {
            int v=(c&0x01);
            result.push_back(v);
            c=(c>>1);
        }
    }

    //11->22 remove continue 1
    for(int i=1;i<result.size();i++)
    {
        if(result[i]==1&&result[i-1]==1)
        {
            result[i]=2;
            result[i-1]=2;
        }
    }

    //2222->2122
    int size=result.size();
    for(int i=1;i<size;i++)
    {
        if(result[i-1]==2&&result[i]==2&&result[(i+1)%size]==2&&result[(i+2)%size]==2)
        {
            result[i]=1;
            i+=3;
        }
    }

    //00000->01110
    for(int i=2;i<size-2;i++)
    {
        if(result[i-1]==0&&result[i-2]==0&&result[i]==0&&result[i+1]==0&&result[i+2]==0)
        {
            result[i-1]=1;
            result[i]=1;
            result[i+1]=1;
            i+=4;
        }
    }


    //set end flag
    result[size-1]=0;
    result[size-2]=1;
    result[size-3]=1;
    result[size-4]=2;

    return result;
}

void drawCircle(cv::Mat &codeImage,const std::vector<int> &drawCodeVector,int radius,const cv::Scalar &color,float offset)
{
    assert(drawCodeVector.size()%8==0);
    int begin=0;
    for(int i=1;i<drawCodeVector.size();i++)
    {
        if(drawCodeVector[i]!=drawCodeVector[begin])
        {
            drawSections(codeImage,radius,drawCodeVector.size(),begin,i,drawCodeVector[begin],color,offset);
            begin=i;
        }
    }
    drawSections(codeImage,radius,drawCodeVector.size(),begin,drawCodeVector.size(),drawCodeVector[begin],color,offset);
}

void drawCodeOnLine(cv::Mat &codeImage,const std::vector<int> &drawCode, cv::Point2f p1,cv::Point2f p2, const cv::Scalar &color,int spaceEnd)
{
//    cv::line(codeImage,p1,p2,cv::Scalar(0,0,255),2);
    const int centerX=codeImage.cols/2;
    const int centerY=codeImage.rows/2;
//    return;
    int size=drawCode.size();
    int splitSize=size+spaceEnd;
    float xStep=(p2.x-p1.x)/splitSize;
    float yStep=(p2.y-p1.y)/splitSize;
    float slide=sqrt(xStep*xStep+yStep*yStep);

    float xSlide=xStep/slide;
    float ySlide=yStep/slide;
    const float LINE_WIDTH=DRAW_SEGMENT_WIDTH;
    const float width_bias_x=LINE_WIDTH*xSlide;
    const float width_bias_y=LINE_WIDTH*ySlide;
    float segmentStartX=-1;
    float segmentStartY=-1;
    float segmentCnt=0;

    p1.x+=(xStep*spaceEnd/2);
    p1.y+=(yStep*spaceEnd/2);
    p2.x-=(xStep*spaceEnd/2);
    p2.y-=(yStep*spaceEnd/2);

    for(int i=0;i<size;i++)
    {
        int code=drawCode[i];
        float x=p1.x+xStep*i;
        float y=p1.y+yStep*i;
        if(code==EMPTY)
        {
            if(segmentCnt!=0)
            {
                float nx=x-width_bias_x;
                float ny=y-width_bias_y;
                cv::line(codeImage,cv::Point2f(segmentStartX,segmentStartY),cv::Point(nx,ny),color,LINE_WIDTH,cv::LINE_AA);
//                cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(p1.x+xStep*i,p1.y+yStep*i),cv::Scalar(0,0,255));
//                cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(segmentStartX-width_bias_x,segmentStartY-width_bias_y),cv::Scalar(0,255,0));
                segmentCnt=0;
            }

            continue;
        }


        if(code==POINT)
        {
            cv::circle(codeImage,cv::Point2f(x+(xStep/2),y+(yStep/2)),DRAW_SEGMENT_WIDTH,color,-1,cv::LINE_AA);
            if(segmentCnt!=0)
            {
                float nx=x-width_bias_x;
                float ny=y-width_bias_y;
                cv::line(codeImage,cv::Point2f(segmentStartX,segmentStartY),cv::Point(nx,ny),color,LINE_WIDTH,cv::LINE_AA);
//                cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(p1.x+xStep*i,p1.y+yStep*i),cv::Scalar(0,0,255));
//                cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(segmentStartX-width_bias_x,segmentStartY-width_bias_y),cv::Scalar(0,255,0));
                segmentCnt=0;
            }
        }
        else if(code==SEGMENT)
        {

            float px=x+width_bias_x;
            float py=y+width_bias_y;

            if(segmentCnt==0)
            {
                segmentStartX=px;
                segmentStartY=py;

            }
            segmentCnt++;

        }
    }
    if(segmentCnt!=0)
    {
        float nx=p2.x-width_bias_x;
        float ny=p2.y-width_bias_y;
        cv::line(codeImage,cv::Point2f(segmentStartX,segmentStartY),cv::Point(nx,ny),color,LINE_WIDTH,cv::LINE_AA);
//        cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(p2.x,p2.y),cv::Scalar(0,0,255));
//        cv::line(codeImage,cv::Point2f(centerX,centerY),cv::Point2f(segmentStartX-width_bias_x,segmentStartY-width_bias_y),cv::Scalar(0,255,0));
        segmentCnt=0;
    }
}

void drawHexagonSegment(cv::Mat &codeImage,const std::vector<int> &drawCodeVector,int radius,int startIndex,const cv::Scalar &color,float offset,int spaceEnd)
{
    const int centerX=codeImage.cols/2;
    const int centerY=codeImage.rows/2;

    const float sectionLen=M_PI/3;
    const float beginRadian=startIndex*sectionLen+offset;
    const float endRadian=(startIndex+1)*sectionLen+offset;

    float x1=radius*cos(beginRadian)+centerX;
    float y1=-radius*sin(beginRadian)+centerY;

    float x2=radius*cos(endRadian)+centerX;
    float y2=-radius*sin(endRadian)+centerY;

    std::vector<int> tmp;
//    tmp.push_back(EMPTY);
    for(int i=0;i<8;i++)
    {
        tmp.push_back(drawCodeVector[startIndex*8+i]);
    }

//    tmp.push_back(EMPTY);

    if(spaceEnd>=2)
    {
        tmp[0]=0;
//        if(tmp[7]==1)
            tmp[7]=0;
        float x1CStep=x1-centerX;
        float y1CStep=y1-centerY;

        float xc=centerX+(x1CStep*0.75);
        float yc=centerY+(y1CStep*0.75);

        int splitSize=tmp.size()+spaceEnd;
        float x12Step=(x2-x1)/splitSize;
        float y12Step=(y2-y1)/splitSize;
        float xt=x1+x12Step;
        float yt=y1+y12Step;
        float r=sqrt((xt-xc)*(xt-xc)+(yt-yc)*(yt-yc));

        cv::ellipse(codeImage,cv::Point2f(xc,yc),cv::Size(r,r),0,
                    -radianToAngle(beginRadian-DRAW_STEP)+30,
                    -radianToAngle(beginRadian+DRAW_STEP)-30,
                    color,DRAW_SEGMENT_WIDTH,cv::LINE_AA);
//        if(startIndex%2==1)
//        {
//            cv::ellipse(codeImage,cv::Point2f(xc,yc),cv::Size(r,r),0,
//                        -radianToAngle(beginRadian-DRAW_STEP)+10,
//                        -radianToAngle(beginRadian+DRAW_STEP)-10,
//                        cv::Scalar(255,255,255),DRAW_SEGMENT_WIDTH,cv::LINE_AA);
//        }
    }
    drawCodeOnLine(codeImage,tmp,cv::Point(x1,y1),cv::Point(x2,y2),color,spaceEnd);

}

void drawHexagon(cv::Mat &codeImage,const std::vector<int> &drawCodeVector,int radius,const cv::Scalar &color,float offset,int spaceEnd)
{
    int byteNum=drawCodeVector.size()/8;
    assert(drawCodeVector.size()%8==0);
    for(int i=0;i<byteNum;i++)
    {
        drawHexagonSegment(codeImage,drawCodeVector,radius,i,color,offset,spaceEnd);
    }

}

std::vector<int> beautifyCode(const std::vector<int> &drawCode)
{
    std::vector<int> result;
    result.reserve(drawCode.size());
    int size=drawCode.size();
    for(int i=0;i<size;i++)
    {
        if(drawCode[i]==0)
        {
            result.push_back(2);
        }
        else
            result.push_back(drawCode[i]);
    }
    for(int i=1;i<size;i++)
    {
        if(result[i-1]==2&&result[i]==2&&result[(i+1)%size]==2)
        {
            result[i]=1;
            i+=2;
        }
    }
    return result;
}

bool code2Image(cv::Mat &codeImage,const uchar code[],int length,const cv::Scalar &color)
{
    const int codeLength=TOTAL_SECTION/8;
    if(length>codeLength-1)
        return false;

    uchar codeWidthFlag[codeLength];
    memcpy(codeWidthFlag,code,sizeof(uchar)*length);
    codeWidthFlag[5]=(uchar)(length<<4);

    std::vector<int> drawCodeVector=generateSectionCode(codeWidthFlag,codeLength);
    drawCircle(codeImage,drawCodeVector,MAX_RADIUS,color,0);
    drawCircle(codeImage,drawCodeVector,MIN_RADIUS,color,M_PI);
//    drawHexagon(codeImage,drawCodeVector,MIN_RADIUS,color,M_PI,1);


    drawCodeVector=beautifyCode(drawCodeVector);
    drawHexagon(codeImage,drawCodeVector,MAX_RADIUS-43,color,-M_PI_2,2);
    drawHexagon(codeImage,drawCodeVector,MAX_RADIUS-8,color,M_PI,2);


    return true;
}
inline bool hitTest(const cv::Mat& codeImageBinary,int row,int col)
{
    int cnt=0;
    for(int i=-1;i<=1;i++)
    {
        for(int j=-1;j<=1;j++)
        {
            if(codeImageBinary.at<uchar>(row+i,col+j)!=255)
                cnt++;
        }
    }
    if(cnt>=4)
        return true;
    else
        return false;
}
int valueInSection(const cv::Mat& codeImageBinary,float radius,int totalSections,int index,float offset,const int hit_threshold[][2])
{
    const int centerX=codeImageBinary.cols/2;
    const int centerY=codeImageBinary.rows/2;

    const float sectionLen=M_PI*2/totalSections;
    const float beginRadian=index*sectionLen+offset;
    const float endRadian=(index+1)*sectionLen+offset;

    const float step=1.0/MAX_RADIUS;

    int maxHitCnt=0;
    int hitCnt=0;
    for(float radian=beginRadian+step;radian<endRadian-step;radian+=step)
    {
        int x=radius*cos(radian)+centerX;
        int y=-radius*sin(radian)+centerY;
        if(hitTest(codeImageBinary,y,x))
        {
            hitCnt++;
        }
        else
        {
            if(hitCnt>maxHitCnt)
                maxHitCnt=hitCnt;
            hitCnt=0;
        }

    }
    if(hitCnt>maxHitCnt)
        maxHitCnt=hitCnt;
    for(int i=0;i<2;i++)
    {
        if(maxHitCnt>=hit_threshold[i][0]&&maxHitCnt<=hit_threshold[i][1])
            return (i+1);
    }
    return 0;

}

bool parseSectionCode(const std::vector<int>&sectionCode,int codeLength,uchar code[])
{
    if(sectionCode.size()<codeLength*8)
        return false;
    for(int i=0;i<codeLength;i++)
    {
        uchar c=0;
        for(int j=0;j<8;j++)
        {
            int index=i*8+j;
            int value=sectionCode[index];
            if(value>1)
                value=1;
            c+=(value<<j);
        }
        code[i]=c^MASK;
    }
    return true;
}

bool parseCircleCodeInImage(const cv::Mat& codeImage,float radius,int length,uchar code[],float offset,const int hit_threshold[][2])
{
    std::vector<int> sectionCodes;
    sectionCodes.reserve(TOTAL_SECTION);
    for(int i=0;i<TOTAL_SECTION;i++)
    {
        int v=valueInSection(codeImage,radius,TOTAL_SECTION,i,offset,hit_threshold);
        sectionCodes.push_back(v);
    }
    //check end flag
//    int e1=sectionCodes[TOTAL_SECTION-1];
    int e2=sectionCodes[TOTAL_SECTION-2];
    int e3=sectionCodes[TOTAL_SECTION-3];
//    int e4=sectionCodes[TOTAL_SECTION-4];
    if(e2!=1||e3!=1)
        return false;

    //01110->00000
    for(int i=1;i<TOTAL_SECTION-1;i++)
    {
        if(sectionCodes[i-1]==1&&sectionCodes[i]==1&&sectionCodes[i+1]==1)
        {
            sectionCodes[i-1]=0;
            sectionCodes[i]=0;
            sectionCodes[i+1]=0;
            i+=2;
        }
    }

    bool ret= parseSectionCode(sectionCodes,length,code);
    if(ret)
        return true;
}
std::vector<int> parseLineCode(const cv::Mat& codeImage,int radius,int index,float offset)
{
    std::vector<int> result;
    const int centerX=codeImage.cols/2;
    const int centerY=codeImage.rows/2;
    const float sectionLen=M_PI/3;
    const float beginRadian=index*sectionLen+offset;
    const float endRadian=(index+1)*sectionLen+offset;

    float x1=radius*cos(beginRadian)+centerX;
    float y1=-radius*sin(beginRadian)+centerY;

    float x2=radius*cos(endRadian)+centerX;
    float y2=-radius*sin(endRadian)+centerY;

    int splitSize=8+1;
    float xStep=(x2-x1)/splitSize;
    float yStep=(y2-y1)/splitSize;
    float slide=sqrt(xStep*xStep+yStep*yStep);
    float xSlide=xStep/slide;
    float ySlide=yStep/slide;

    for(int i=0;i<8;i++)
    {
        float x=x1+(i+0.5)*xStep;
        float y=y1+(i+0.5)*yStep;
        float nx=x1+(i+1.5)*xStep;
        float ny=y1+(i+1.5)*yStep;
        int hitCnt=0;
        int maxHitCnt=0;
        while(x<nx&&y<ny)
        {
            if(hitTest(codeImage,y,x))
            {
                hitCnt++;
            }
            else
            {
                if(hitCnt>maxHitCnt)
                    maxHitCnt=hitCnt;
                hitCnt=0;
            }

            x+=xSlide;
            y+=ySlide;
        }
        if(hitCnt>maxHitCnt)
            maxHitCnt=hitCnt;
        bool flag=false;
        for(int i=0;i<2;i++)
        {
            if(maxHitCnt>=CODE_HIT_INTERVAL_MIN[i][0]&&maxHitCnt<=CODE_HIT_INTERVAL_MIN[i][1])
            {
                result.push_back(i+1);
            }
        }
        if(!flag)
        {
            result.push_back(0);
        }
    }
    return result;
}


bool parseHexagon(const cv::Mat& codeImage,int length,uchar code[],float offset)
{
    std::vector<int> sectionCodes;
    sectionCodes.reserve(TOTAL_SECTION);
    for(int i=0;i<6;i++)
    {
         std::vector<int> lineCode= parseLineCode(codeImage,MIN_RADIUS, i,offset);
         for(int c:lineCode)
             sectionCodes.push_back(c);
    }


}

bool image2Code(const cv::Mat& codeImage,int length,uchar code[],int informationLength)
{
    unsigned char decodeInfo[informationLength];
    for(float offset=0;offset<M_PI*2;offset+=M_PI_2)
    {
        if(parseCircleCodeInImage(codeImage,MAX_RADIUS, length,code,offset,CODE_HIT_INTERVAL_MAX))
        {
            bool crcAns = decodeCRCAndRS(code, length, informationLength, decodeInfo);
            if(crcAns)
                return true;
        }
        if(parseCircleCodeInImage(codeImage,MIN_RADIUS,length,code,offset+M_PI,CODE_HIT_INTERVAL_MIN))
        {
            bool crcAns = decodeCRCAndRS(code, length, informationLength, decodeInfo);
            if(crcAns)
                return true;
        }
    }
    return false;
}
