#include <iostream>
#include <string>
using namespace std;

extern void rotateBMP(char *szInfile, char *szOutFile, float fTheta);
extern void rotateColors(char *szInfile, char *szOutFile);
extern void sobel(char *szInfile, char *szOutFile);
extern void gaussian(char *szInfile, char *szOutFile, int size);
extern void sumtest(int n);
extern void heat(const char *szFile, int size);

int main(int argc, char **argv)
{
    try
    {
        //rotateBMP(argv[1], argv[2], atof(argv[3]));
        //rotateColors(argv[1], argv[2]);
        sobel(argv[1], argv[2]);
        //gaussian(argv[1], argv[2], atoi(argv[3]));
        //sumtest(atoi(argv[1]));

        //heat(argv[1], atoi(argv[2]));

    }
    catch(string &s)
    {
        cout << "Error : " << s << endl;
    }

    cout << "Hit it to quit it..." << endl;
    getchar();
    
    return 0;
}


