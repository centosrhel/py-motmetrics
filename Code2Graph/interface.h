#include <opencv2/core.hpp>
//编码，目的是添加CRC验证字段和RS纠错码
//input 需要编码的信息
//inputLength input内元素个数
//output 输出的编码
//output内元素的个数
void encodeCRCAndRS(unsigned char input[],int inputLength,unsigned char output[],int &outputLength);

//解码，解码成功返回true,否则返回false
//input 编码后的信息
//inputLenght input内元素的个数
//informationLength input内实际信息元素个数,剩余的部分为CRC和RS码
//output 解码出来的信息
bool decodeCRCAndRS(unsigned char input[],int inputLength,int informationLength,unsigned char output[]);

//读取图像中的编码
//codeImage, 承载信息的图像，CV_UINT8_C1,值为255表示空白0，其他数值均表示1；
//length, 需要解读的信息的长度
//code, 输出结果
//informationLength
//返回true表示读取成功，否则表示读取失败

bool image2Code(const cv::Mat& codeImage,int length,uchar code[],int informationLength=2);
//将解码后的信息绘制到图像上
//codeImage, 画布(BGR）
//code 信息编码
//length code的长度
//color 画笔的颜色
//返回true表示编码成功，否则表示失败
bool code2Image(cv::Mat &codeImage,const uchar code[],int length,const cv::Scalar &color);
