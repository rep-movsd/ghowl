#include <iostream>
#include "ghowl.h"

// Sum adjacent elements 
KERNEL void kSum(int *p, int n)
{
    int i;
    getLinearThreadIndex(i);
    if(i < n)
    {
        int a = p[i * 2], b = p[i * 2 + 1];
        p[i * 2] = p[i * 2 + 1] = 0;
        p[i] = a + b;
    }
}
//////////////////////////////////////////////////////////////////////////

void sumtest(int n)
{
    int N = n;
    try
    {
        // Allocate host array (size must be an even number initially so add 1 extra if needed)
        if(n % 2) n++;
        TCPUArray<int> h1(n);

        // Initialize it
        int *ph = h1.ptr();
        for(int i = 0; i < N; ++i)
        {
            ph[i] = i;
        }
        // allocate array on GPU and get contents into it
        TGPUArray<int> d1(h1);
        
        while(n > 1)
        {
            int bOdd = n % 2;
            n /= 2;
            if(bOdd) n++;

            cerr << "Summing " << n << endl;
            int nBlocks, nThreadsPerBlock;
            getBlockGridSizes(n, nBlocks, nThreadsPerBlock);
            KERNEL_1D_GRID_1D(kSum, nBlocks, nThreadsPerBlock)(d1.ptr(), n);
        }

        // Get device memory back to host (ideally we need to copy only 1st element)
        h1 = d1;

        cout << "Sum = " << ph[0] << endl;
        cout << "(N * N-1) / 2 = " << ((N * (N-1))/2) << endl;
    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}

