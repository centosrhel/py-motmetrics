#include <opencv2/opencv.hpp>
using namespace std;


int main()
{
    std::stringstream ss;
    int index=0;
    for(int i=0;i<285;i++)
    {
        ss.str(std::string());
        ss<<"/home/sixd-ailabs/Develop/QRCode/ChladniImages/"<<i<<".jpg";
        cv::Mat img=cv::imread(ss.str());
        if(img.empty())
            continue;
        ss.str(std::string());
        ss<<"/home/sixd-ailabs/Develop/QRCode/ChladniImagesRename/"<<index++<<".jpg";
        cv::imwrite(ss.str(),img);

    }

    return 0;

}
