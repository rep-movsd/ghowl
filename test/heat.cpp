#include <iostream>
#include "ghowl.h"

// Both CUDA functions are idempotent, so extra threads executing this code will not cause trouble, 
// This allows us to avoid conditionals
__global__ void kHeatFlow(TPitchPtr ps, int y)
{
    T2DView<byte> pArr(ps);

    // get X co ordinate for this thread ( avoid edgemost pixels )
    int x;
    getLinearThreadIndex(x);
    x %= (ps.w - 2);
    x += 1;

    // assign (x, y + 1) to be average of (x-1, y) (x, y) and (x+1, y)
    float fAvg = (pArr[y][x-1] + pArr[y][x] + pArr[y][x+1]) / 3.0f;
    pArr[y+1][x] = (byte)fAvg;
}
//////////////////////////////////////////////////////////////////////////


void heat(const char *szFile, int size)
{
    try
    {
        TCPUArray<byte> temp(size, size);
        // Fill the first row of array with a random pattern
        FOR(i, size)
        {
            temp.p2d()[0][i] = rand() % 256;
        }


        TGPUArray<byte> d(temp);

        // Apply heat flow 
        int nBlocks, nThreadsPerBlock;
        getBlockGridSizes(size-2, nBlocks, nThreadsPerBlock);
        for(int i = 0; i < size-1; ++i)
        {
            KERNEL_1D_GRID_1D(kHeatFlow, nBlocks, nThreadsPerBlock)(d.data(), i);
        }

        // Assign device memory to bitmap and save it to output file
        TBitmap<byte> bmp(d);
        bmp.save(szFile);
    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}

