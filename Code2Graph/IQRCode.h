//
//  IQRCode.h
//
//  Created by Hu Yafei on 23/03/2018.
//

#ifndef IQRCode_h
#define IQRCode_h

#include <memory>

namespace Ali_QRCode
{
#define NORMALIZED_WIDTH 430 //标准QRCode图像宽度
#define NORMALIZED_HEIGHT 430 //标准QRCode图像高度
#define NORMALIZED_nCHANNELS 3 //标准QRCode图像通道数
//#define NORMALIZED_SIZE \
//((NORMALIZED_WIDTH+2) * (NORMALIZED_HEIGHT+2) * NORMALIZED_nCHANNELS)
#define NORMALIZED_SIZE \
(NORMALIZED_WIDTH * NORMALIZED_HEIGHT * NORMALIZED_nCHANNELS)

struct Ali_Image
{
    unsigned char* data;
    unsigned int width, height, nChannels;
};

class IQRCode
{
public:
    virtual ~IQRCode(){}

    /*!
     * 输入：ID, QRCode对应的ID，当前支持ushort(0～65535)
     * 输出：outputImgData, ID对应的QRCode图像的data指针
     * 返回值： -1，找不到所需的Chladni图案文件； -2，对ID编码发生错误； 0，成功生成QRCode
     */
    virtual int generateQRCode(unsigned long ID, unsigned char* outputImgData) = 0;

    /*!
     * 输入：ali_image, 待识别的QRCode图像，当前支持BGR三通道图像
     * 输出：ID, 识别结果，当前范围为ushort(0～65535)
     * 返回值： -1，定位失败，ID无效； -2，image2Code失败，ID无效；-3，decodeCRCAndRS失败，ID无效；0，识别成功，ID有效；
     */
    virtual int recognizeQRCode(const Ali_Image& ali_image, unsigned long& ID) = 0;
};

/*!
 * 输入：strDirectory, Chladni图案所在的目录；num_pictures, 此目录中Chladni图案的数目，图像命名格式为(0～(num_pictures-1)).jpg;
 * 输出：
 * 返回值：IQRCode*
 */
//IQRCode* createIQRCode(const char* strDirectory, const unsigned num_pictures);
std::shared_ptr<IQRCode> makeIQRCode(const char* strDirectory, const unsigned num_pictures);

//void releaseIQRCode(IQRCode**);
} // end namespace Ali_QRCode
#endif /* IQRCode_h */
