#include "ghowl.h"

// Kernel function combines 2 arays into the R and G channels of the third
KERNEL void kCombine(TPitchPtr p1, TPitchPtr p2, TPitchPtr p3)
{
    // Which pixel is this thread operating on?
    TXY pt;
    getThreadXY(pt, p1.w, p1.h);

    // Arrays of red green and blue planes
    // for input and output images
    T2DView<float> d1(p1);
    T2DView<float> d2(p2);

    T2DView<float> ro(p3, 0);
    T2DView<float> go(p3, 1);
    T2DView<float> bo(p3, 2);

    ro[pt] = d1[pt];
    go[pt] = d2[pt];
    bo[pt] = 0;
}
///////////////////////////////////////////////////////

void sobel(char *szInfile, char *szOutFile)
{
    try
    {
        // Make the 2 filter arrays
        float vsobel[3][3] = { {+3, +10, +3}, {0, 0, 0}, {-3, -10, -3} };
        float hsobel[3][3] = { {+3, 0, -3}, {+10, 0, -10}, {+3, 0, -3} };
        TGPUArray<float> vFilter, hFilter;
        vFilter.assign2dArr<3, 3>(vsobel);
        hFilter.assign2dArr<3, 3>(hsobel);
        
        // Load a bmp
        TBitmap<float> bmp;
        bmp.load(szInfile, true);

        // Load bmp data into device buffer, and create one for the output
        TGPUArray<float> src(bmp), dst1(bmp.dims()), dst2(bmp.dims());

        // Apply horz and vertical filters
        applyImageFilter(src, dst1, vFilter);
        applyImageFilter(src, dst2, hFilter);

        // Combine the two, red and green for vert and horz
        TGPUArray<float> dst3(bmp.width(), bmp.height(), 3);

        int nElems = bmp.height() * bmp.width();
        int nBlocks, nThreads;
        getBlockGridSizes(nElems, nBlocks, nThreads);

        KERNEL_1D_GRID_1D(kCombine, nBlocks, nThreads)(dst1.data(), dst2.data(), dst3.data());

        // Assign device memory back to bitmap and save it to output file
        TBitmap<float> bmp2(dst3);
        bmp2.save(szOutFile);

    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}


