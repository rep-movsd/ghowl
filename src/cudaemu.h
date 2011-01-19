#define GPU_CPU __device__ __host__ 
#define GPU __device__

#ifndef __CUDACC__

#define cudaSuccess 0;

#define cudaHostAllocDefault        0   ///< Default page-locked allocation flag
#define cudaHostAllocPortable       1   ///< Pinned memory accessible by all CUDA contexts
#define cudaHostAllocMapped         2   ///< Map allocation into device space
#define cudaHostAllocWriteCombined  4   ///< Write-combined memory

#define cudaEventDefault            0   ///< Default event flag
#define cudaEventBlockingSync       1   ///< Event uses blocking synchronization

#define cudaDeviceScheduleAuto      0   ///< Device flag - Automatic scheduling
#define cudaDeviceScheduleSpin      1   ///< Device flag - Spin default scheduling
#define cudaDeviceScheduleYield     2   ///< Device flag - Yield default scheduling
#define cudaDeviceBlockingSync      4   ///< Device flag - Use blocking synchronization
#define cudaDeviceMapHost           8   ///< Device flag - Support mapped pinned allocations
#define cudaDeviceMask              0xf ///< Device flags mask

#define __global__
#define __host__
#define __device__
#define __constant__
#define __shared__ static
#define __sharedlocal__  
#define KERNEL 

//////////////////////////////////////////////////////////////////////////

typedef int cudaError_t;
typedef int cudaStream_t;

struct cudaArray;
struct cudaChannelFormatDesc;

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
};

struct dim3
{
    dword x, y, z;
    dim3(dword x = 1, dword y = 1, dword z = 1)
        : x(x), y(y), z(z) {}
};

struct cudaPitchedPtr
{
    void   *ptr;      ///< Pointer to allocated memory
    dword  pitch;    ///< Pitch of allocated memory in bytes
    dword  xsize;    ///< Logical width of allocation in elements
    dword  ysize;    ///< Logical height of allocation in elements
};

struct cudaExtent
{
    dword width;     ///< Width in bytes
    dword height;    ///< Height in bytes
    dword depth;     ///< Depth in bytes

    inline bool operator==(const cudaExtent &other) const
    {
        return width == other.width &&
            height == other.height &&
            depth == other.depth;
    }
};

struct cudaPos
{
    dword x;
    dword y;
    dword z;
};

struct cudaMemcpy3DParms
{
    struct cudaArray     *srcArray;
    struct cudaPos        srcPos;
    struct cudaPitchedPtr srcPtr;
    struct cudaArray     *dstArray;
    struct cudaPos        dstPos;
    struct cudaPitchedPtr dstPtr;
    struct cudaExtent     extent;
    enum cudaMemcpyKind   kind;
};

struct cudaDeviceProp
{
    char   name[256];                 ///< ASCII string identifying device
    size_t totalGlobalMem;            ///< Global memory available on device in bytes
    size_t sharedMemPerBlock;         ///< Shared memory available per block in bytes
    int    regsPerBlock;              ///< 32-bit registers available per block
    int    warpSize;                  ///< Warp size in threads
    size_t memPitch;                  ///< Maximum pitch in bytes allowed by memory copies
    int    maxThreadsPerBlock;        ///< Maximum number of threads per block
    int    maxThreadsDim[3];          ///< Maximum size of each dimension of a block
    int    maxGridSize[3];            ///< Maximum size of each dimension of a grid
    int    clockRate;                 ///< Clock frequency in kilohertz
    size_t totalConstMem;             ///< Constant memory available on device in bytes
    int    major;                     ///< Major compute capability
    int    minor;                     ///< Minor compute capability
    size_t textureAlignment;          ///< Alignment requirement for textures
    int    deviceOverlap;             ///< Device can concurrently copy memory and execute a kernel
    int    multiProcessorCount;       ///< Number of multiprocessors on device
    int    kernelExecTimeoutEnabled;  ///< Specified whether there is a run time limit on kernels
    int    integrated;                ///< Device is integrated as opposed to discrete
    int    canMapHostMemory;          ///< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
    int    computeMode;               ///< Compute mode (See ::cudaComputeMode)
    int    __cudaReserved[36];
};

extern dim3 threadIdx, blockIdx, blockDim, gridDim;

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Macros for kernel invocation
//////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(X) ;

// One dimensional block of threads - N is an integer
#define KERNEL_1D(f, N)                                     \
    gridDim.x = gridDim.y = gridDim.z = 1;                  \
    blockDim.x = N; blockDim.y = blockDim.z = 1;            \
    blockIdx.x = blockIdx.y = blockIdx.z = 0;               \
    threadIdx.x = threadIdx.y = threadIdx.z = 0;             \
    for(; threadIdx.x < N; ++threadIdx.x) f

// N dimensional block of threads - D is a dim3
#define KERNEL_ND(f, D)                                             \
    gridDim.x = gridDim.y = gridDim.z = 1;                          \
    blockDim = D;                                                   \
    blockIdx.x = blockIdx.y = blockIdx.z = 0;                       \
    threadIdx.x = threadIdx.y = threadIdx.z = 0;                    \
    for(threadIdx.z = 0; threadIdx.z < D.z; ++threadIdx.z)          \
    for(threadIdx.y = 0; threadIdx.y < D.y; ++threadIdx.y)      \
    for(threadIdx.x = 0; threadIdx.x < D.x; ++threadIdx.x)  \
    f

// 1D block of threads with a 1D grid of blocks - G & N are integers
#define KERNEL_1D_GRID_1D(f, G, N)                              \
    gridDim.x = G; gridDim.y = gridDim.z = 1;                   \
    blockDim.x = N; blockDim.y = blockDim.z = 1;                \
    blockIdx.x = blockIdx.y = blockIdx.z = 0;                   \
    threadIdx.x = threadIdx.y = threadIdx.z = 0;                \
    for(blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)           \
    for(threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) f


// ND block of threads with a 1D grid of blocks - G is int D is dim3
#define KERNEL_ND_GRID_1D(f, G, D)                              \
    gridDim.x = G; gridDim.y = gridDim.z = 1;                   \
    blockDim = D;                                              \
    blockIdx.x = blockIdx.y = blockIdx.z = 0;                \
    for(blockIdx.x = 0; blockIdx.x < G; ++blockIdx.x)           \
    for(threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)  \
    for(threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)  \
    for(threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)  \
    f

// 1D block of threads with a 3D grid of blocks - D is int G is dim3
#define KERNEL_1D_GRID_ND(f, G, N)                          \
    gridDim = G;                                           \
    blockDim.x = N; blockDim.y = blockDim.z = 1;            \
    for(blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z)     \
    for(blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)     \
    for(blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)     \
    for(threadIdx.x = threadIdx.y = threadIdx.z = 0;    \
    threadIdx.x < N; ++threadIdx.x)\
    f

#define KERNEL_ND_GRID_ND(f, G, N)                              \
    gridDim.x = gridDim.y = gridDim.z = G;                   \
    blockDim.x = blockDim.y = blockDim.z = N;                \
    for(blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z)         \
    for(blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y)         \
    for(blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x)         \
    for(threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z)  \
    for(threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y)  \
    for(threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x)  \
    f

//////////////////////////////////////////////////////////////////////////


cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeArray(cudaArray *array);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
cudaError_t cudaGetSymbolSize(dword *size, const char *symbol);
cudaError_t cudaHostAlloc(void **ptr, dword size, dword flags);
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, dword flags);
cudaError_t cudaHostGetFlags(dword *pFlags, void *pHost);
cudaError_t cudaMalloc(void **devPtr, dword size);
cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, cudaExtent extent);
cudaError_t cudaMalloc3DArray(cudaArray **arrayPtr, const cudaChannelFormatDesc *desc, cudaExtent extent);
cudaError_t cudaMallocArray(cudaArray **arrayPtr, const cudaChannelFormatDesc *desc, dword width, dword height);
cudaError_t cudaMallocHost(void **ptr, dword size);
cudaError_t cudaMallocPitch(void **devPtr, dword *pitch, dword width, dword height);
cudaError_t cudaMemcpy(void *dst, const void *src, dword count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2D(void *dst, dword dpitch, const void *src, dword spitch, dword widthInBytes, dword height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DArrayToArray(cudaArray *dst, dword wOffsetDst, dword hOffsetDst, const cudaArray *src, dword wOffsetSrc, dword hOffsetSrc, dword width, dword height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void *dst, dword dpitch, const void *src, dword spitch, dword width, dword height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DFromArray(void *dst, dword dpitch, const cudaArray *src, dword wOffset, dword hOffset, dword width, dword height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, dword dpitch, const cudaArray *src, dword wOffset, dword hOffset, dword width, dword height, enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy2DToArray(cudaArray *dst, dword wOffset, dword hOffset, const void *src, dword spitch, dword width, dword height, cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *dst, dword wOffset, dword hOffset, const void *src, dword spitch, dword width, dword height, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p);
cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p, cudaStream_t stream);
cudaError_t cudaMemcpyArrayToArray(cudaArray *dst, dword wOffsetDst, dword hOffsetDst, const cudaArray *src, dword wOffsetSrc, dword hOffsetSrc, dword count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, dword count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromArray(void *dst, const cudaArray *src, dword wOffset, dword hOffset, dword count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src, dword wOffset, dword hOffset, dword count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol, dword count, dword offset, cudaMemcpyKind kind);
cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, dword count, dword offset, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToArray(cudaArray *dst, dword wOffset, dword hOffset, const void *src, dword count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyToArrayAsync(cudaArray *dst, dword wOffset, dword hOffset, const void *src, dword count, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, dword count, dword offset, cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src, dword count, dword offset, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaMemset(void *devPtr, int value, dword count);
cudaError_t cudaMemset2D(void *devPtr, dword pitch, int value, dword width, dword height);
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent);
cudaError_t cudaSetDeviceFlags(int flags);
cudaError_t cudaGetLastError();
cudaError_t cudaThreadSynchronize();
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int iDev);

int atomicAdd(int* address, int val);

#ifdef _MSC_VER
    __declspec(deprecated("__syncthreads() will not work in serial CPU mode!!")) void __syncthreads();
#else
#define __syncthreads "__syncthreads() will not work in serial CPU mode!!"
#endif



#else

#define __sharedlocal__ __shared__
#define KERNEL __global__

#define CHECK_CUDA(X) g_e.setContext(X) ; g_e = cudaGetLastError()

#define KERNEL_1D(f, N)  f <<< 1, N >>>   
#define KERNEL_ND(f, D)  f <<< 1, D >>>   

#define KERNEL_1D_GRID_1D(f, G, N)  f <<< G, N >>> 
#define KERNEL_1D_GRID_ND(f, G, N)  f <<< G, N >>>
#define KERNEL_ND_GRID_1D(f, G, N)  f <<< G, N >>>
#define KERNEL_ND_GRID_ND(f, G, N)  f <<< G, N >>>

#endif
