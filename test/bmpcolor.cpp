#include "ghowl.h"

// Example kernel function to simply rotate the colors r->g g->b b->r
__global__ void kFilter(TPitchPtr ps, TPitchPtr pd)
{
    // Which pixel is this thread operating on?
    TXY pt;
    getThreadXY(pt, ps.w, ps.h);

    // Arrays of red green and blue planes
    // for input and output images
    T2DView<byte> ri(ps, 0);
    T2DView<byte> gi(ps, 1);
    T2DView<byte> bi(ps, 2);
    T2DView<byte> ro(pd, 0);
    T2DView<byte> go(pd, 1);
    T2DView<byte> bo(pd, 2);

    ro[pt] = gi[pt];
    go[pt] = bi[pt];
    bo[pt] = ri[pt];
}
///////////////////////////////////////////////////////

void rotateColors(char *szInfile, char *szOutFile)
{
    try
    {
        // Load a bmp
        TBitmap<byte> bmp;
        bmp.load(szInfile);

        // Load bmp data into device buffer, and create one for the output
        TGPUArray<byte> src(bmp), dst(bmp.dims());

        // Calculate blocks and threads
        int nThreadsPerBlock, nBlock;
        getBlockGridSizes(bmp.width() * bmp.height(), nBlock, nThreadsPerBlock);

        // Rotate image in device buffer by fTheta
        KERNEL_1D_GRID_1D(kFilter, nBlock, nThreadsPerBlock)(src.data(), dst.data());

        // Assign device memory back to bitmap and save it to output file
        bmp = dst;
        bmp.save(szOutFile);
    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}
