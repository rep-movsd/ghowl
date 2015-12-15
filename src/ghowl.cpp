#include "ghowl.h"

TCudaDeviceProp g_devProps;
TCudaException g_e;

#ifndef REGION_CUDA_EMU_

#ifndef __CUDACC__

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
using std::map;

//////////////////////////////////////////////////////////////////////////
// Functions that emulate the CUDA API in plain ol' C++
//////////////////////////////////////////////////////////////////////////

dim3 threadIdx, blockIdx, blockDim, gridDim;
const int g_iAlign = 256;
map<void*, void*> g_AlignedPtrs;

//void __syncthreads()
//{
//    assert(!"__syncthreads() will not work in C++ mode!!");
//}
////////////////////////////////////////////////////////////////////////////

int atomicAdd(int* address, int val) 
{ 
    int old = *address; 
    *address += val;  
    return old; 
}
//////////////////////////////////////////////////////////////////////////

int getAlignedValue(int i, int align = g_iAlign)
{
    if(i % g_iAlign) 
    {
        i += g_iAlign;
        i -= i % g_iAlign;
    }

    return i;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMalloc(void**p, dword iSize)
{
    *p = calloc(1, iSize);
    //    cerr << "Allocated(" << int(*p) << ")" << endl;
    return p ? 0 : 2;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaHostAlloc(void **p, dword iSize, dword flags)
{
    *p = calloc(1, iSize);
    return p ? 0 : 2;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaFree(void *p)
{
    if(g_AlignedPtrs.count(p))
    {
        //cerr << "Freeing(" << int(g_AlignedPtrs[p]) << ")" << endl;
        free(g_AlignedPtrs[p]);
        g_AlignedPtrs.erase(p);
    }
    else
    {
        //cerr << "Freeing(" << int(p) << ")" << endl;
        free(p);
    }

    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMallocPitch(void **devPtr, dword *pitch, dword w, dword h)
{
    // Ensure width of each row is a multiple of align and >= width
    *pitch = getAlignedValue(w);

    // Allocate extra bytes so that the buffer pointer can be aligned
    dword cbBuf = *pitch * h + g_iAlign;
    *devPtr = calloc(1, cbBuf);
    if(!*devPtr)
        return 2;

    // Ensure buffer pointer is a multiple of align and >= width
    int iAligned = getAlignedValue((*(int*)(*devPtr)));

    // Save the unaligned actual ptr so we can free it later
    g_AlignedPtrs[(void*)iAligned] = *devPtr;

    //cerr << "Allocated(" << int(*devPtr) << ") (" << iAligned << ")" << endl;

    *devPtr = (void*)iAligned;
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMalloc3D(cudaPitchedPtr *p, cudaExtent e)
{
    p->xsize = e.width;
    p->ysize = e.height;

    return cudaMallocPitch(&p->ptr, &p->pitch, e.width, e.height * e.depth);
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMemcpy(void *d, const void* s, dword n, cudaMemcpyKind kind)
{
    memcpy(d, s, n);
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, dword count, dword offset, cudaMemcpyKind kind)
{
    memcpy((void*)(symbol + offset), src, count);
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, dword count, dword offset, cudaMemcpyKind kind)
{
    memcpy(dst, (void*)(symbol + offset), count);
    return 0;
}
//////////////////////////////////////////////////////////////////////////


cudaError_t cudaMemcpy2D(void *dst, dword dpitch, const void *src, dword spitch, dword widthInBytes, dword height, cudaMemcpyKind kind)
{
    char *d = (char*)dst, *s = (char*)src;
    for(dword y = 0; y < height; ++y)
    {
        memcpy(d, s, widthInBytes);
        d += dpitch;
        s += spitch;
    }

    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *q)
{
    cudaMemcpy3DParms parms = *q, *p = &parms;

    char *pDst = (char*)p->dstPtr.ptr;
    char *pSrc = (char*)p->srcPtr.ptr;

    for(dword z = 0; z < p->extent.depth; ++z)
    {
        cudaMemcpy2D(pDst, p->dstPtr.pitch, pSrc, p->srcPtr.pitch, 
            p->extent.width, p->extent.height, 
            cudaMemcpyHostToHost);

        pDst += p->dstPtr.pitch * p->extent.height;
        pSrc += p->srcPtr.pitch * p->extent.height;
    }

    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaMemset(void *devPtr, int value, dword count)
{
    memset(devPtr, value, count);
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaSetDeviceFlags(int flags)
{
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, dword flags)
{
    *pDevice = pHost;
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetLastError()
{
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaThreadSynchronize()
{
    return 0;
}
//////////////////////////////////////////////////////////////////////////

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int iDev)
{
    strcpy(prop->name, "GeForce GTX 285");   ///< ASCII string identifying device
    prop->totalGlobalMem = 256 * 1048576;   ///< Global memory available on device in bytes
    prop->sharedMemPerBlock = 16384;        ///< Shared memory available per block in bytes
    prop->regsPerBlock = 16384;             ///< 32-bit registers available per block
    prop->warpSize = 32;                    ///< Warp size in threads
    prop->memPitch = 262144;                ///< Maximum pitch in bytes allowed by memory copies
    prop->maxThreadsPerBlock = 512;         ///< Maximum number of threads per block
    prop->maxThreadsDim[0] = 512;                 ///< Maximum size of each dimension of a block
    prop->maxThreadsDim[1] = 512;                 ///< Maximum size of each dimension of a block
    prop->maxThreadsDim[2] = 64;                 ///< Maximum size of each dimension of a block


    prop->maxGridSize[0] = 65536;                   ///< Maximum size of each dimension of a grid
    prop->maxGridSize[1] = 65536;                   ///< Maximum size of each dimension of a grid
    prop->maxGridSize[2] = 1;                   ///< Maximum size of each dimension of a grid

    prop->clockRate;                        ///< Clock frequency in kilohertz
    prop->totalConstMem = 65536;            ///< Constant memory available on device in bytes
    prop->major = 1;                        ///< Major compute capability
    prop->minor = 3;                        ///< Minor compute capability
    prop->textureAlignment = 256;           ///< Alignment requirement for textures
    prop->deviceOverlap = 1;                ///< Device can concurrently copy memory and execute a kernel
    prop->multiProcessorCount = 30;         ///< Number of multiprocessors on device
    prop->kernelExecTimeoutEnabled = 1;     ///< Specified whether there is a run time limit on kernels
    prop->integrated = 0;                   ///< Device is integrated as opposed to discrete
    prop->canMapHostMemory = 1;             ///< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    prop->computeMode = 0;                  ///< Compute mode (See ::cudaComputeMode)

    return 0;
}

#else

#define CHECK_CUDA(X) g_e.setContext(X) ; g_e = cudaGetLastError()

#define KERNEL_1D(f, N)  f <<< 1, N >>>   
#define KERNEL_ND(f, D)  f <<< 1, D >>>   

#define KERNEL_1D_GRID_1D(f, G, N)  f <<< G, N >>> 
#define KERNEL_1D_GRID_ND(f, G, N)  f <<< G, N >>>
#define KERNEL_ND_GRID_1D(f, G, N)  f <<< G, N >>>
#define KERNEL_ND_GRID_ND(f, G, N)  f <<< G, N >>>

#endif

#endif

// Image filter kernel
__global__ void kApplyImageFilter1D(TPitchPtr dataIn, TPitchPtr dataOut, float *pFilter, int iFilterSize)
{
    //T2DClipArr<float> pData(dataIn, 0, dataIn.w - 1, 0, dataIn.h - 1);
    T2DDefView<float> pData(dataIn, 0, dataIn.w, 0, dataIn.h);
    TXY ptThis;
    getThreadXY(ptThis, dataIn.w, dataIn.h);

    // filtersize negative means vertical else horizontal
    bool bVertical = false;
    if(iFilterSize < 0)
    {
        iFilterSize = -iFilterSize;
        bVertical = true;
    }

    // Get midpoint of filter matrix
    int iMid = iFilterSize / 2;

    // Apply the filter
    float sum = 0, count = 0;
    TXY ptImg = ptThis;

    if(bVertical)
    {
        ptImg.y -= iMid;
        FOR(i, iFilterSize)
        {
            float f = pFilter[i];
            sum += pData[ptImg] * f;
            count += f;
            ptImg.y++;        
        }
    }
    else
    {
        ptImg.x -= iMid;
        FOR(i, iFilterSize)
        {
            float f = pFilter[i];
            sum += pData[ptImg] * f;
            count += f;
            ptImg.x++;        
        }
    }

    if(count == 0) count = 1;
    float val = sum / count;
    clampval(val, 0.0f, 255.0f);

    T2DView<float> pDataOut(dataOut);
    pDataOut[ptThis] = val;
}
//////////////////////////////////////////////////////////////////////////


// Image filter kernel
__global__ void kApplyImageFilter(TPitchPtr dataIn, TPitchPtr dataOut, TPitchPtr filter)
{
    T2DView<float> pFilter(filter);
    T2DDefView<float> pData(dataIn, 0, dataIn.w, 0, dataIn.h);

    TXY ptThis;
    getThreadXY(ptThis, dataIn.w, dataIn.h);

    // Get midpoint of filter matrix
    TXY ptHalf(filter.w, filter.h);
    ptHalf /= 2;

    // Apply the filter
    float sum = 0, count = 0;
    TXY ptFilter;
    LOOP_PT_YX(ptFilter, filter.w, filter.h)
    {
        TXY ptImg = ptThis + (ptFilter - ptHalf);
        float f = pFilter[ptFilter];
        sum += pData[ptImg] * f;
        count += f;
    }

    if(count == 0) count = 1;
    float val = sum / count;
    clampval(val, 0.0f, 255.0f);

    T2DView<float> pDataOut(dataOut);
    pDataOut[ptThis] = val;
}
//////////////////////////////////////////////////////////////////////////

std::string TCudaException::errorMessage( int e )
{
    static const char *sCudaErrorMessages[] = 
    {
        "No errors",
        "Missing configuration",
        "Memory allocation",
        "Initialization",
        "Launch failure",
        "Prior launch failure",
        "Launch timeout",
        "Launch out of resources",
        "Invalid device function",
        "Invalid configuration",
        "Invalid device",
        "Invalid value",
        "Invalid pitch value",
        "Invalid symbol",
        "Map buffer object failed",
        "Unmap buffer object failed",
        "Invalid host pointer",
        "Invalid device pointer",
        "Invalid texture",
        "Invalid texture binding",
        "Invalid channel descriptor",
        "Invalid memcpy direction",
        "Address of constant",
        "Texture fetch failed",
        "Texture not bound",
        "Synchronization error",
        "Invalid filter setting",
        "Invalid norm setting",
        "Mixed device execution",
        "CUDA runtime unloading",
        "Unknown error condition",
        "Function not yet implemented",
        "Memory value too large",
        "Invalid resource handle",
        "Not ready",
        "CUDA runtime is newer than driver",
        "Set on active process",
        "No available CUDA device",
    };

    if(e >= 0 && e <= 38) 
        return sCudaErrorMessages[e];
    else if(e == 0x7f)
        return "Startup failure";
    else if(e >= 1000)
        return "API error";

    return "Unknown error";
}
//////////////////////////////////////////////////////////////////////////

void TCudaException::doThrow() const
{
    cerr << message() << endl;
    throw *this;
}
//////////////////////////////////////////////////////////////////////////

void TCudaException::operator=( int e )
{
    if(e)
    {
        err = e;
        doThrow();
    }
}
//////////////////////////////////////////////////////////////////////////

std::string TCudaException::message() const
{
    ostringstream oss;
    oss << sContext << " failed with code " << err << " : " <<  errorMessage(err);
    return oss.str();
}
//////////////////////////////////////////////////////////////////////////

void TCudaException::setContext( pcchar sCtx )
{
    sContext = sCtx;
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// CUDA thread helpers
//////////////////////////////////////////////////////////////////////////



bool dimsEqual(const cudaExtent &e1, const cudaExtent &e2)
{
    return e1.width == e2.width && e1.height == e2.height && e1.depth == e2.depth;
}
//////////////////////////////////////////////////////////////////////////


void applyImageFilter(TArray<float> &pDataIn, TArray<float> &pDataOut, TArray<float> &pFilter)
{
    ENSURE_THAT(pFilter.data().bOnDevice == pDataIn.data().bOnDevice, "Filter and input data matrix must reside on device xor host memory");
    ENSURE_THAT(pFilter.data().bOnDevice == pDataOut.data().bOnDevice, "Filter and output data matrix must reside on device xor host memory");
    ENSURE_THAT(pFilter.getShape() == 2 && pDataIn.getShape() == 2 && pDataOut.getShape() == 2, "Filters and data matrix must be 2 dimensional");
    ENSURE_THAT(dimsEqual(pDataIn.dims(), pDataOut.dims()), "Filters and data matrix must be 2 dimensional");

    int nElems = pDataIn.height() * pDataIn.width();
    int nBlocks, nThreads;
    getBlockGridSizes(nElems, nBlocks, nThreads);

    g_e.setContext("kApplyImageFilter");
    KERNEL_1D_GRID_1D(kApplyImageFilter, nBlocks, nThreads)(pDataIn.data(), pDataOut.data(), pFilter.data());
    g_e = cudaThreadSynchronize();
    
}
//////////////////////////////////////////////////////////////////////////

void applyImageFilterFast(TArray<float> &pDataIn, TArray<float> &pDataOut, TArray<float> &pTemp, TArray<float> &pFilter)
{
    ENSURE_THAT(pFilter.data().bOnDevice == pDataIn.data().bOnDevice, "Filter and input data matrix must reside on device xor host memory");
    ENSURE_THAT(pFilter.data().bOnDevice == pDataOut.data().bOnDevice, "Filter and output data matrix must reside on device xor host memory");
    ENSURE_THAT(pFilter.data().bOnDevice == pTemp.data().bOnDevice, "Filter and temp data matrix must reside on device xor host memory");

    ENSURE_THAT(pFilter.getShape() == 2 && pDataIn.getShape() == 2 && pDataOut.getShape() == 2, "Filters and data matrix must be 2 dimensional");
    ENSURE_THAT(dimsEqual(pDataIn.dims(), pDataOut.dims()), "Filters and data matrix must be 2 dimensional");
    ENSURE_THAT(pFilter.height() == 2 && pDataIn.getShape() == 2 && pDataOut.getShape() == 2, "Filters and data matrix must be 2 dimensional");

    int nElems = pDataIn.height() * pDataIn.width();
    int nBlocks, nThreads;
    getBlockGridSizes(nElems, nBlocks, nThreads);

    T2DView<float> pArrFilter(pFilter.data());

    g_e.setContext("kApplyImageFilter1D_1");
    KERNEL_1D_GRID_1D(kApplyImageFilter1D, nBlocks, nThreads)(pDataIn.data(), pTemp.data(), pArrFilter[0], -pFilter.width());
    g_e = cudaThreadSynchronize();

    g_e.setContext("kApplyImageFilter1D_2");
    KERNEL_1D_GRID_1D(kApplyImageFilter1D, nBlocks, nThreads)(pTemp.data(), pDataOut.data(), pArrFilter[1], pFilter.width());
    g_e = cudaThreadSynchronize();
}
//////////////////////////////////////////////////////////////////////////


