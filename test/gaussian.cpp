#include "ghowl.h"

void gaussian(char *szInfile, char *szOutFile, int size)
{
    try
    {
        // Load a bmp
        TBitmap<float> bmp;
        bmp.load(szInfile, true);

        // Load bmp data into device buffer, and create one for the output
        // One extra temporary one for the filter function
        TGPUArray<float> src(bmp), dst(bmp.dims()), tmp(bmp.dims());

        // Init a CPU array with a gaussian filter ( just the central row and the central column )
        TCPUArray<float> g(size, 2);
        getGaussianKernel(size/2, &(g.p2d()[0][0]));
        getGaussianKernel(size/2, &(g.p2d()[1][0]));

        // Put the filter into a GPU array
        TGPUArray<float> dg(g);
        applyImageFilterFast(src, dst, tmp, dg);

        // Assign device memory back to bitmap and save it to output file
        bmp = dst;
        bmp.save(szOutFile, true);

    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}

