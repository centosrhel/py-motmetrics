#include <iostream>
#include "interface.h"
using namespace std;

int main()
{
    unsigned char in[2] = {'1', 'd'};
    unsigned char out[5];
    unsigned char info[2];
    int outLen = 0, infoLen = 2;

    cout << "input data: ";
    for(int i = 0; i < infoLen; i++){
        cout << (int)in[i] << " ";
    }
    cout << endl;
    encodeCRCAndRS(in, 2, out, outLen);

    cout << "encoded data: ";
    for(int i = 0; i < outLen; i++){
        cout << (int)out[i] << " ";
    }
    cout << endl;
    // add an error
    out[1] = 'p';

    cout << "errorous data: ";
    for(int i = 0; i < outLen; i++){
        cout << (int)out[i] << " ";
    }
    cout << endl;

    bool crcAns = decodeCRCAndRS(out, outLen, infoLen, info);
    if(crcAns){
        cout << "CRC: Right answer!" << endl;
    }else{
        cout << "CRC: Wrong answer!" << endl;
    }

    cout << "recovered data: ";
    for(int i = 0; i < infoLen; i++){
        cout << (int)info[i] << " ";
    }
    cout << endl;
    return 0;
}
