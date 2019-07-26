//
//  QRCode.h
//  RS_CRC_Code
//
//  Created by Hu Yafei on 23/03/2018.
//

#ifndef QRCode_h
#define QRCode_h

#include "IQRCode.h"
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace Ali_QRCode
{
class QRCode : public IQRCode
{
public:
    QRCode(const char* strDirectory, const unsigned num_pictures);
    ~QRCode(){}
    virtual int generateQRCode(unsigned long ID, unsigned char* outputImgData);
    virtual int recognizeQRCode(const Ali_Image& ali_image, unsigned long& ID);
private:
    const std::string m_strDirectory;
    const unsigned m_num_pictures;
    std::vector<std::vector<cv::Point> > m_contours;
    cv::Scalar m_color;
    //std::vector<std::vector<cv::Point> > m_contours2;
};
} // end namespace Ali_QRCode

#endif /* QRCode_h */
