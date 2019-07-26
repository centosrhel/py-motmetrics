//
//  IQRCode.cpp
//
//
//  Created by Hu Yafei on 23/03/2018.
//

#include "QRCode.h"
//using namespace Ali_QRCode;

namespace Ali_QRCode{
//IQRCode* createIQRCode(const char* strDirectory, const unsigned num_pictures)
std::shared_ptr<IQRCode> makeIQRCode(const char* strDirectory, const unsigned num_pictures)
{
//    IQRCode* pObj = new QRCode(strDirectory, num_pictures);
//    return pObj;
    return std::shared_ptr<IQRCode>(new QRCode(strDirectory, num_pictures));
}

//void releaseIQRCode(IQRCode** pIQRCode)
//{
//    if (pIQRCode && *pIQRCode)
//    {
//        delete (*pIQRCode);
//        *pIQRCode = nullptr;
//    }
//}
}
