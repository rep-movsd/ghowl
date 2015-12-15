#ifndef __CUDALIB_H__
#define __CUDALIB_H__


#ifdef _MSC_VER

#ifndef GHOWL_BUILD
#ifdef _DEBUG
    #pragma comment(lib, "ghowl_d.lib")
#else
    #pragma comment(lib, "ghowl.lib")
#endif
#endif
#endif

#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <set>
#include <map>
#include <stddef.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Short typedefs and forward declares
//////////////////////////////////////////////////////////////////////////
typedef unsigned int dword;
typedef unsigned char byte;
typedef unsigned short word;
typedef char *pchar;
typedef const char *pcchar;
typedef void** ppv;

template<typename T> struct TCoOrd2D;
template<typename T> struct TCoOrd3D;

typedef TCoOrd2D<int> TXY;
typedef TCoOrd2D<float> TFXY;

typedef TCoOrd3D<int> TXYZ;
typedef TCoOrd3D<float> TFXYZ;

template<int N, class T> class TMany;
template<typename T> struct T2DView;
template<typename T> struct T3DView;

// CUDA Emulation stuff
#include "cudaemu.h"

///////////////////////////////////////////////////////////////////////////
// Macros for looping 
///////////////////////////////////////////////////////////////////////////
#define LOOP_PT_YX(PT, W, H)               \
    for(PT.y = 0; PT.y < H; ++PT.y)    \
    for(PT.x = 0; PT.x < W; ++PT.x)


#define LOOP_YX(W, H)               \
    for(int y = 0; y < H; ++y)    \
    for(int x = 0; x < W; ++x)

#define LOOP_XY(W, H)           \
    for(int x = 0; x < W; ++x) \
    for(int y = 0; y < H; ++y)  

#define LOOP_ZYX(W, H, D)\
    for(int z = 0; z < D; ++z)\
    for(int y = 0; y < H; ++y)\
    for(int x = 0; x < W; ++x)

#define FOR(I, N) for(int I = 0; I < ((int)N); ++I)

#define LOOP_YX_PAD(W, H, P)               \
    for(int y = P; y < H - P; ++y)    \
    for(int x = P; x < W - P; ++x)


#define FOR_EACH(T, I, C) for(T::iterator i = C.begin(); i != C.end(); ++i)

#define ALL(X) X.begin(), X.end()

///////////////////////////////////////////////////////////////////////////
// Macros for frequently used trigonometric constants
///////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979323F
#define COS60 0.5F
#define SIN60 0.86602540378443864676372317075294F
#define SIN30 COS60
#define COS30 SIN60
#define TAN30 (SIN30/COS30)
#define COT30 (COS30/SIN30)
#define TAN60 (SIN60/COS60)
#define COT60 (COS60/SIN60)

//////////////////////////////////////////////////////////////////////////
// CUDA kernel code does not manage to do copy construction so we need to 
// roll our own. 
//////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__
    #define DECL_CUDA_COPY_CTOR(X) GPU_CPU X(const X& that) { *this = that; } 
#else
    #define DECL_CUDA_COPY_CTOR(X) ;
#endif

// Assertion macro
#define ENSURE_THAT(X, Y) if(!(X)) throw (string("Assertion failure : ") + #Y + "\nExpression : " + #X);

///////////////////////////////////////////////////////////////////////////
// TCudaException converts CUDA error codes to C++ exceptions
// Assigning an integer to a TCudaException object will cause it to 
// get thrown
//////////////////////////////////////////////////////////////////////////
class TCudaException
{
    string sContext;
    int err;

    void doThrow() const;
    static string errorMessage(int e);

public:
    void operator=(int e);          // Assignment from integer ( typically a result from a CUDA API call )
    string message() const;         // Get the error message
    void setContext(pcchar sCtx);   // Set error context 
};
//////////////////////////////////////////////////////////////////////////

// Helper class to retrieve device parameters
struct TCudaDeviceProp : public cudaDeviceProp
{
    TCudaDeviceProp()
    {
        cudaGetDeviceProperties(this, 0);
    }
};

// Global TCudaException variable which can be assigned to the return values of CUDA API calls
extern TCudaException g_e;
extern TCudaDeviceProp g_devProps;

//////////////////////////////////////////////////////////////////////////

//CUDA friendly analogs of the STL numeric functors defined via macros
#define DECLARE_BINARY_FUNCTOR(NAME, OP) \
    template<typename T>  struct NAME                   \
    {                                                   \
        GPU_CPU const T operator()(const T o1, const T o2)      \
        {                                               \
            return o1 OP o2;                            \
        }                                               \
    }
//////////////////////////////////////////////////////////////////////////

DECLARE_BINARY_FUNCTOR(opadd, +);
DECLARE_BINARY_FUNCTOR(opsub, -);
DECLARE_BINARY_FUNCTOR(opmul, *);
DECLARE_BINARY_FUNCTOR(opdiv, /);
DECLARE_BINARY_FUNCTOR(opmod, %);

/////////////////////////////////////////////////////////////////////////
// TCoOrd3D and TCoOrd2D represent 3D and 2D co ordinates
/////////////////////////////////////////////////////////////////////////
template<typename T> struct TCoOrd2D
{
    T x, y;
    typedef T Type;

    DECL_CUDA_COPY_CTOR(TCoOrd2D);

    // Default ctor
    GPU_CPU TCoOrd2D() : x(), y()
    {
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct from x and y
    GPU_CPU TCoOrd2D(T ax, T ay) : x(ax), y(ay)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    template<class T2>
    GPU_CPU TCoOrd2D(T2 ax, T2 ay) : x(T(ax)), y(T(ay))
    {
    }
    //////////////////////////////////////////////////////////////////////////

    //Construct from any type with x and y members
    template<typename T2> GPU_CPU TCoOrd2D(const T2 &p)
    {
        *this = p;
    }
    //////////////////////////////////////////////////////////////////////////

    //Assign from any type with x and y members
    template<typename T2> GPU_CPU TCoOrd2D& operator=(const T2 &p)
    {
        x = (T)p.x;
        y = (T)p.y;
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////

    // Y major comparision operator ( 1000000 may be too less )
    GPU_CPU bool operator <(const TCoOrd2D &o) const
    {
        int i = y * 1000000 + x;
        int j = o.y * 1000000 + o.x;
        return i < j;
//         if(y != o.y)
//             return y < o.y;
//         return x < o.x;
    }
    //////////////////////////////////////////////////////////////////////////

    // Generic function that applies op to x and y with ox and oy
    template< template<class> class TOperator > GPU_CPU void op(T ox, T oy)
    {
        TOperator<T> o;
        x = o(x, ox);
        y = o(y, oy);
    }
    //////////////////////////////////////////////////////////////////////////

    // Dot product
    GPU_CPU T dot(TCoOrd2D &o)
    {
        return (T)(o.x * x + o.y * y);
    }
    //////////////////////////////////////////////////////////////////////////
    
    GPU_CPU T dist(TCoOrd2D &o)
    {
        return sqrtf(o.x * x + o.y * y);
    }
    //////////////////////////////////////////////////////////////////////////

    // Rotate co-ordinate
    GPU_CPU void rotate(float fTheta)
    {
        float fC = cos(fTheta), fS = sin(fTheta);
        T nx = x * fC + y * fS;
        T ny = y * fC - x * fS;
        x = nx; y = ny;
    }
    //////////////////////////////////////////////////////////////////////////

    // returns if x is in [x1, x2) and y is in [y1,y2)
    GPU_CPU bool isBoundedBy(T x1, T y1, T x2, T y2)
    {
        return (x >= x1 && x < x2 && y >= y1 && y < y2);
    }
    //////////////////////////////////////////////////////////////////////////

    GPU_CPU void clamp(T minx, T maxx, T miny, T maxy)
    {
        clampval(x, minx, maxx);
        clampval(y, miny, maxy);
    }
    //////////////////////////////////////////////////////////////////////////

    GPU_CPU void wrap(T maxx, T maxy)
    {
        op<opadd>(maxx, maxy);
        op<opmod>(maxx, maxy);
    }
    //////////////////////////////////////////////////////////////////////////
    
    // Sets this point halfway between this and the other one
    template<typename T2> GPU_CPU void averageWith(const T2 &o)
    {
        op<opadd>(o.x, o.y);
        op<opdiv>(2.0, 2.0);
    }
    //////////////////////////////////////////////////////////////////////////

    // Operators defined with macros
    #define DECL_OPEQ_OPS(OP) \
        template<typename T2> GPU_CPU inline TCoOrd2D &operator OP (const TCoOrd2D<T2> &o) { x OP (T2)o.x; y OP (T2)o.y; return *this; } \
        template<typename TOperand> GPU_CPU inline TCoOrd2D &operator OP (const TOperand &o) { x OP o; y OP o; return *this; }

    #define DECL_BINARY_OPS(OP) \
        template<typename T2> GPU_CPU inline TCoOrd2D operator OP (const TCoOrd2D<T2> &o) const { return TCoOrd2D(x OP (T2)o.x, y OP (T2)o.y);} \
        template<typename TOperand> GPU_CPU inline TCoOrd2D operator OP (const TOperand &o) const { return TCoOrd2D(x OP o, y OP o);}

    GPU_CPU inline bool operator ==(const TCoOrd2D &o) const { return o.x == x && o.y == y; }
    GPU_CPU inline bool operator !=(const TCoOrd2D &o) const { return !(o == *this); }
    
    DECL_OPEQ_OPS(+=)
    DECL_OPEQ_OPS(-=)
    DECL_OPEQ_OPS(*=)
    DECL_OPEQ_OPS(/=)
    DECL_OPEQ_OPS(%=)
    DECL_OPEQ_OPS(&=)
    DECL_OPEQ_OPS(|=)
    DECL_OPEQ_OPS(^=)

    DECL_BINARY_OPS(+)
    DECL_BINARY_OPS(-)
    DECL_BINARY_OPS(*)
    DECL_BINARY_OPS(/)
    DECL_BINARY_OPS(%)
    DECL_BINARY_OPS(&)
    DECL_BINARY_OPS(|)
    DECL_BINARY_OPS(^)

    #undef DECL_EQUALITY_OP
    #undef DECL_OPEQ_OPS
    #undef DECL_BINARY_OPS
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct TCoOrd3D
{
    T x, y, z;
    typedef T Type;

    DECL_CUDA_COPY_CTOR(TCoOrd3D);

    // Default ctor
    GPU_CPU TCoOrd3D() : x(0), y(0), z(0)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct from x and y
    GPU_CPU TCoOrd3D(T ax, T ay, T az) : x(ax), y(ay), z(az)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    //Construct from any type with x, y and z members
    template<typename T2> GPU_CPU explicit TCoOrd3D(const T2 &p)
    {
        *this = p;
    }
    //////////////////////////////////////////////////////////////////////////

    //Assign from any type with x, y and z members
    template<typename T2> GPU_CPU TCoOrd3D& operator=(const T2 &p)
    {
        x = (T)p.x;
        y = (T)p.y;
        z = (T)p.z;
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////

    // comparision operator 
    GPU_CPU bool operator <(const TCoOrd3D &o) const
    {
        if(z != o.z) 
            return z < o.z;

        if(y != o.y)
            return y < o.y;

        return x < o.x;
    }
    //////////////////////////////////////////////////////////////////////////

    // Generic function that applies op to x and y with ox and oy
    template< template<class> class TOperator > GPU_CPU void op(T ox, T oy, T oz)
    {
        TOperator<T> o;
        x = o(x, ox);
        y = o(y, oy);
        z = o(z, oz);
    }
    //////////////////////////////////////////////////////////////////////////

    // Dot product
    GPU_CPU T dot(TCoOrd3D &o)
    {
        return (T)(o.x * x + o.y * y + o.z * z);
    }
    //////////////////////////////////////////////////////////////////////////

    // Sets this point halfway between this and the other one
    template<typename T2> GPU_CPU void averageWith(const T2 &o)
    {
        op<opadd>(o.x, o.y, o.z);
        op<opdiv>(2.0, 2.0, 2.0);
    }
    //////////////////////////////////////////////////////////////////////////

    // Operators defined with macros
    #define DECL_EQUALITY_OP(OP) GPU_CPU bool operator OP (const TCoOrd3D &o) const { return o.x OP x && o.y OP y && o.z OP z; }


    #define DECL_OPEQ_OP(OP) \
    template<typename T2> GPU_CPU TCoOrd3D &operator OP (const TCoOrd3D<T2> &o) { x OP o.x; y OP o.y; z OP o.z; return *this;} \
    template<typename TOperand> GPU_CPU TCoOrd3D &operator OP (const TOperand &o) { x OP o; y OP o; z OP o; return *this;}

    #define DECL_BINARY_OP(OP) \
    template<typename T2> GPU_CPU TCoOrd3D operator OP (const TCoOrd3D<T2> &o)  const { return TCoOrd3D(x OP o.x, y OP o.y, z OP o.z);} \
    template<typename TOperand> GPU_CPU TCoOrd3D operator OP (const TOperand &o) const { return TCoOrd3D(x OP o, y OP o, z OP o);}

    DECL_EQUALITY_OP(==)
    DECL_EQUALITY_OP(!=)

    DECL_OPEQ_OP(+=)
    DECL_OPEQ_OP(-=)
    DECL_OPEQ_OP(*=)
    DECL_OPEQ_OP(/=)
    DECL_OPEQ_OP(%=)
    DECL_OPEQ_OP(&=)
    DECL_OPEQ_OP(|=)
    DECL_OPEQ_OP(^=)

    DECL_BINARY_OP(+)
    DECL_BINARY_OP(-)
    DECL_BINARY_OP(*)
    DECL_BINARY_OP(/)
    DECL_BINARY_OP(%)
    DECL_BINARY_OP(&)
    DECL_BINARY_OP(|)
    DECL_BINARY_OP(^)

    #undef DECL_BINARY_OP
    #undef DECL_OPEQ_OP
    #undef DECL_EQUALITY_OP
};
//////////////////////////////////////////////////////////////////////////

// Special class to help create arrays of structs in CUDA
// Beware!! No construction is done!!
// Plain arrays of structs don't work in kernel code if struct has a ctor
template<int N, class T> class TMany
{
    enum{ TypeSize = sizeof(T) };
    enum{ AlignSize = TypeSize % 64 ? (TypeSize + 64) & 63 : TypeSize };
    enum{ Size = N * TypeSize };
    byte pts[Size];

public:
    int size;  // convenient place to hold the size 

    GPU_CPU TMany()
    {
        size = 0;
    }

    GPU_CPU T &operator[](int n)
    {
        return *((T*)(pts + (n * AlignSize)));
    }

    GPU_CPU void push(const T &val)
    {
        (*this)[size] = val;
        size++;
    }
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Helpers for [] operator support on pitched buffers
//////////////////////////////////////////////////////////////////////////

struct TPitchPtr
{
    TPitchPtr(){};
    DECL_CUDA_COPY_CTOR(TPitchPtr);
    char *ptr;
    int pitch;    ///< Pitch of allocated memory in bytes
    int w;    ///< Logical width of allocation in elements
    int h;    ///< Logical height of allocation in elements
    int d;
    bool bOnDevice;

    template<typename T>  
    GPU_CPU T3DView<T> p3d() {return T3DView<T>(*this);}
    
    template<typename T>  
    GPU_CPU T2DView<T> p2d(int iz = 0) {return T2DView<T>(*this, iz);}
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct T2DView : public TPitchPtr
{
    DECL_CUDA_COPY_CTOR(T2DView);

    GPU_CPU T2DView() {};

    GPU_CPU T2DView(const TPitchPtr &p, int z = 0): TPitchPtr(p)
    {
        ptr += pitch * h * z;
    }

    GPU_CPU T& operator[](const TXY &p) const
    {
        return ((T*)(ptr + pitch * p.y))[p.x];
    }

    GPU_CPU T *operator[](int iy) const
    {
        return (T*)(ptr + iy * pitch);
    }
};
//////////////////////////////////////////////////////////////////////////

// "Clamped" array - clamps out of range indexes to specified limits
template<typename T> struct T2DClipView : public TPitchPtr
{
private:
    T2DClipView(){};
    
public:
    int minx, miny, maxx, maxy;
    DECL_CUDA_COPY_CTOR(T2DClipView);

    GPU_CPU T2DClipView(const TPitchPtr &p, int aminx, int amaxx, int aminy, int amaxy, int z = 0): 
        TPitchPtr(p), minx(aminx), maxx(amaxx), miny(aminy), maxy(amaxy)  
    {
        ptr += pitch * h * z;
    };

    GPU_CPU T& operator[](TXY p) const
    {
        p.clamp(minx, maxx, miny, maxy);
        return ((T*)(ptr + pitch * p.y))[p.x];
    }
};
//////////////////////////////////////////////////////////////////////////

// "Default" array - returns a specified value for out of bound indices
template<typename T> struct T2DDefView : public TPitchPtr
{
private:
    T2DDefView(){};
    T defaultVal;

public:
    int minx, maxx, miny, maxy;
    DECL_CUDA_COPY_CTOR(T2DDefView);

    GPU_CPU T2DDefView(const TPitchPtr &p, int aminx, int amaxx, int aminy, int amaxy, T aDefaultVal = T(), int z = 0): 
    TPitchPtr(p), minx(aminx), maxx(amaxx), miny(aminy), maxy(amaxy)
    {
        defaultVal = aDefaultVal;
        ptr += pitch * h * z;
    };

    GPU_CPU T operator[](TXY p) const
    {
        if(p.x < minx || p.x >= maxx || p.y < miny || p.y >= maxy) 
            return defaultVal;
        else
            return ((T*)(ptr + pitch * p.y))[p.x];
    }
};
//////////////////////////////////////////////////////////////////////////

// "Wrapped" array - wraps out of range values by modulus
template<typename T> struct T2DWrapView : public TPitchPtr
{
private:
    T2DWrapView(){};

public:
    int maxx, maxy;
    DECL_CUDA_COPY_CTOR(T2DWrapView);

    GPU_CPU T2DWrapView(const TPitchPtr &p, int amaxx, int amaxy, int z = 0): 
    TPitchPtr(p), maxx(amaxx), maxy(amaxy)  
    {
        ptr += pitch * h * z;
    };

    GPU_CPU T& operator[](TXY p) const
    {
        p.wrap(maxx, maxy);
        return ((T*)(ptr + pitch * p.y))[p.x];
    }
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct T3DView : public TPitchPtr
{
    DECL_CUDA_COPY_CTOR(T3DView);

    GPU_CPU T3DView(const TPitchPtr &p) : TPitchPtr(p)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    GPU_CPU T2DView<T> operator[](int iz)
    {
        return T2DView<T>(*this, iz);
    }
    //////////////////////////////////////////////////////////////////////////

    GPU_CPU T& operator[](const TXYZ &p)
    {
        //assert(p.z > -1 &&  p.w > -1 && p.h > -1 && p.z < d && p.y < h && p.x < w);
        return T2DView<T>(*this, p.z)[p.y][p.x];
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Base class for host and buffer wrapper classes
//////////////////////////////////////////////////////////////////////////
template<typename T> class TArray
{

protected:

    T* m_pBuf;              // Pointer to storage
    int m_cbPitch;        // Number of bytes per row
    cudaExtent m_dim;       // dimensions of this array
    
    // Required only internally for 2d memcpys
    inline cudaPitchedPtr pitchPtr() const
    {
        cudaPitchedPtr p;
        p.pitch = m_cbPitch;
        p.ptr = m_pBuf;
        p.xsize = m_dim.width;
        p.ysize = m_dim.height;
        return p;
    }
    //////////////////////////////////////////////////////////////////////////

    virtual void resize() = 0;
    //////////////////////////////////////////////////////////////////////////

    virtual bool isDeviceBuf() const = 0;
    //////////////////////////////////////////////////////////////////////////
    
    // Compares 2 arrays' dimensions and throws if unequal
    void checkDims(const TArray<T> &that) const
    {
        const cudaExtent &e = that.dims();
        if(depth() != (int)e.depth || height() != (int)e.height || width() != (int)e.width)
            throw string("Dimensions mismatch between buffers");
    }
    //////////////////////////////////////////////////////////////////////////

    void checkShape()
    {
        ENSURE_THAT(m_dim.width && m_dim.height && m_dim.depth, "Cannot resize array to 0 size");

        // We only allow 1x1x1, Mx1x1 MxNx1 or MxNxO sized arrays, not other shapes

        // If width == 1 then other two dims also have to be 1
        ENSURE_THAT(m_dim.width > 1 || (m_dim.height == 1 && m_dim.depth == 1), "Only 1x1x1, Mx1x1 MxNx1 or MxNxO sized arrays allowed");

        // if height == 1 then depth has to be one
        ENSURE_THAT(m_dim.height > 1 || (m_dim.depth == 1), "Only 1x1x1, Mx1x1 MxNx1 or MxNxO sized arrays allowed");
    }
    //////////////////////////////////////////////////////////////////////////
    
    template<typename T_TO, typename T_FROM, cudaMemcpyKind kind>
    void doCopy(const T_TO &to, const T_FROM &from)
    {
        g_e.setContext(__FUNCTION__);
        to.checkDims(from);

        int cbRow = from.width() * from.Size;

        // Check the geometry of the device buffer 
        if(from.depth() > 1) // 3d
        {
            cudaMemcpy3DParms p = {0};
            p.srcPtr = from.pitchPtr();
            p.dstPtr = to.pitchPtr();
            p.extent = from.dims();
            p.extent.width = cbRow;
            p.kind = kind;
            g_e = cudaMemcpy3D(&p);
        }
        else if(from.height() > 1)
        {
            // Check for maximum pitch 
            if(from.pitch() < (int)g_devProps.memPitch && to.pitch() < (int)g_devProps.memPitch)
                g_e = cudaMemcpy2D(to.ptr(), to.pitch(), from.ptr(), from.pitch(), cbRow, to.height(), kind);
            else // memcpy row by row
            {
                char *fromPtr = (char*)from.ptr(), *toPtr = (char*)to.ptr();
                for(int y = 0; y < to.height(); ++y)
                {
                    g_e = cudaMemcpy(toPtr, fromPtr, cbRow, kind);
                    fromPtr += from.pitch();
                    toPtr += to.pitch();
                }
            }
        }
        else
        {
            g_e = cudaMemcpy(to.ptr(), from.ptr(), from.width() * to.Size, kind);
        }
    }
    //////////////////////////////////////////////////////////////////////////

public:

    // size of the element stored
    static const int Size = sizeof(T);
    typedef T Elem;

    // Initially array has no storage and no dimensions
    TArray()
    {
        m_pBuf = 0;
        m_cbPitch = m_dim.width = m_dim.height = m_dim.depth = 0;
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct 1d, 2d or 3d array
    TArray(int ix, int iy = 1, int iz = 1)
    {
        m_dim.width = ix;
        m_dim.height = iy;
        m_dim.depth = iz;
        m_cbPitch = ix * Size;
        m_pBuf = 0;
    }
    //////////////////////////////////////////////////////////////////////////

    virtual ~TArray(){}
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> void assign(const T2 *that)
    {
        // If this array is empty then resize it
        if(!m_pBuf)
        {
            m_dim = that->m_dim;
            resize();
        }

        if(isDeviceBuf())
        {
            if(that->isDeviceBuf())   // Device to device
            {
                doCopy<TArray, T2, cudaMemcpyDeviceToDevice>(*this, *that);
            }
            else                    // Host to device
            {
                doCopy<TArray, T2, cudaMemcpyHostToDevice>(*this, *that);
            }
        }
        else
        {
            if(that->isDeviceBuf())   // Device to Host
            {
                doCopy<TArray, T2, cudaMemcpyDeviceToHost>(*this, *that);
            }
            else                    // Host to Host
            {
                doCopy<TArray, T2, cudaMemcpyHostToHost>(*this, *that);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////

    void setDims(int x, int y = 1, int z = 1)
    {
        m_dim.width = x;
        m_dim.height = y;
        m_dim.depth = z;
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    void setDims(const cudaExtent &e)
    {
        m_dim = e;
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    TPitchPtr data() const
    {
        TPitchPtr p;
        p.pitch = m_cbPitch;
        p.w = m_dim.width;
        p.h = m_dim.height;
        p.d = m_dim.depth;
        p.ptr = (char*)m_pBuf;
        p.bOnDevice = isDeviceBuf();
        return p;
    }
    //////////////////////////////////////////////////////////////////////////

    // Returns the number of dimensions ( dimensions must grow only in x, y, z order)
    int getShape()
    {
        int dim = 0;
        if(m_dim.width > 1) 
            ++dim;

        if(m_dim.height > 1) 
            ++dim;

        if(m_dim.depth > 1) 
            ++dim;

        return dim;
    }
    //////////////////////////////////////////////////////////////////////////

    #define DECL_GETTER(TYPE, FNNAME, RET_EXPR) inline TYPE FNNAME() const { return RET_EXPR; }

    DECL_GETTER(T3DView<T>, p3d, data());
    DECL_GETTER(T2DView<T>, p2d, data());
    DECL_GETTER(T*, ptr, m_pBuf);
    DECL_GETTER(int, width, m_dim.width);
    DECL_GETTER(int, height, m_dim.height);
    DECL_GETTER(int, depth, m_dim.depth);
    DECL_GETTER(int, pitch, m_cbPitch);
    DECL_GETTER(const cudaExtent &, dims, m_dim);

    #undef DECL_GETTER
};
//////////////////////////////////////////////////////////////////////////

// Forward declare
template<typename T> class TGPUArray;
template<typename T> class TCPUArray;
template<typename T> class TBitmap;

//////////////////////////////////////////////////////////////////////////
// Host buffer wrapper class
//////////////////////////////////////////////////////////////////////////
template<typename T> class TCPUArray : public TArray<T>
{
protected:

    vector<char> m_v;

    void resize()
    {
        this->checkShape();

        int ix = this->m_dim.width;
        int iy = this->m_dim.height;
        int iz = this->m_dim.depth;

        // Align on 16 byte boundary
        this->m_cbPitch = ix * this->Size;
        if(this->m_cbPitch % 16)
        {
            this->m_cbPitch &= ~16;
            this->m_cbPitch += 16;
        }

        // can only resize from 0 size 
        //assert(!m_v.size());
        m_v.resize(this->m_cbPitch * iy * iz);
        this->m_pBuf = (T*)&m_v[0];

        T3DView<T> pArr(this->data());
        LOOP_ZYX(ix, iy, iz)
        {
            T* p = &(pArr[z][y][x]);
            new(p) T();
        }
    }
    //////////////////////////////////////////////////////////////////////////

public:

    static const bool isDevice = 0;

    virtual bool isDeviceBuf(void) const 
    { 
        return false; 
    }
    //////////////////////////////////////////////////////////////////////////

    TCPUArray() : TArray<T>()
    {}
    //////////////////////////////////////////////////////////////////////////

    explicit TCPUArray(int ix, int iy = 1, int iz = 1) : TArray<T>(ix, iy, iz)
    {
        this->resize();
    }
    //////////////////////////////////////////////////////////////////////////

    explicit TCPUArray(const cudaExtent &e) : TArray<T>(e.width, e.height, e.depth)
    {
        this->resize();
    }
    //////////////////////////////////////////////////////////////////////////

     // Default copy ctor has to be overridden
    explicit TCPUArray(const TCPUArray &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    explicit TCPUArray(const TArray<T> &that) : TArray<T>()
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    // Default = operator has to be overridden
    void operator=(const TCPUArray &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    void operator=(const TArray<T> &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    template<int M, int N> void assign2dArr(const T pData[M][N])
    {
        this->m_dim.depth = 1;
        this->m_dim.height = M;
        this->m_dim.width = N;
        resize();
        char *pBuf = (char*)this->m_pBuf;
        FOR(y, M)
        {
            std::copy(pData[y], pData[y] + this->width(), (T*)pBuf);
            pBuf += this->pitch();
        }
    }
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Device buffer wrapper class
//////////////////////////////////////////////////////////////////////////
template<typename T> class TGPUArray : public TArray<T>
{
protected:

    // (re)allocates CUDA memory 
    void resize()
    {
        this->checkShape();
        g_e.setContext(__FUNCTION__);
        if(this->m_pBuf)
        {
            g_e = cudaFree(this->m_pBuf);
            this->m_pBuf = 0;
        }

        int ix = this->m_dim.width;
        int iy = this->m_dim.height;
        int iz = this->m_dim.depth;

        if(iz == 1)
        {
            if(iy == 1) // 1D
            {
                T *p;
                g_e = cudaMalloc(ppv(&p), ix * iy * iz * this->Size);
                this->m_pBuf = p;
            }
            else
            {
                int uPitch;
                void *pBuf = this->m_pBuf;
                g_e = cudaMallocPitch(&pBuf, (dword*)&uPitch, ix * this->Size, iy);
                this->m_cbPitch = uPitch;
                this->m_pBuf = (T*)pBuf;
            }
        }
        else // 3D
        {
            cudaPitchedPtr p;
            cudaExtent dim = this->m_dim;
            dim.width *= this->Size;
            g_e = cudaMalloc3D(&p, dim);
            this->m_cbPitch = p.pitch;
            this->m_pBuf = (T*)p.ptr;
        }
    }
    //////////////////////////////////////////////////////////////////////////

public:

    static const bool isDevice = 1;

    virtual bool isDeviceBuf(void) const 
    { 
        return true; 
    }
    //////////////////////////////////////////////////////////////////////////
    
    TGPUArray() : TArray<T>()
    {}
    //////////////////////////////////////////////////////////////////////////

    // Construct 1 2 or 3D buffer given dimensions
    explicit TGPUArray(int ix, int iy = 1, int iz = 1) : TArray<T>(ix, iy, iz)
    {
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    explicit TGPUArray(const cudaExtent &e) : TArray<T>(e.width, e.height, e.depth)
    {
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    // Default copy ctor has to be overridden
    explicit TGPUArray(const TGPUArray &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    explicit TGPUArray(const TArray<T> &that) : TArray<T>()
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    // Default assignment operator has to be overridden
    void operator=(const TGPUArray &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////
    
    void operator=(const TArray<T> &that)
    {
        this->assign(&that);
    }
    //////////////////////////////////////////////////////////////////////////

    template<int M, int N> void assign2dArr(const T pData[M][N])
    {
        TCPUArray<T> temp;
        temp.assign2dArr<M, N>(pData);
        this->assign(&temp);
    }

    // Free allocated memory 
    virtual ~TGPUArray()
    {
        cudaFree(this->m_pBuf);
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// Bitmap handling
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Bitmap class is a 3d array consisting of 1 or 3 planes 
//////////////////////////////////////////////////////////////////////////

template <typename T> class TBitmap : public TCPUArray<T>
{

#ifndef _MSC_VER
        #define PACK_GCC __attribute__((packed))
#else
     #define PACK_GCC
     #pragma pack(push)
#endif

#ifdef _MSC_VER
       #pragma pack(1)
#endif
    struct BMPFILEHEADER
    {
        unsigned short bfType;
        size_t bfSize;
        unsigned short bfReserved1;
        unsigned short bfReserved2;
        size_t bfOffBits;
    } PACK_GCC;
    //////////////////////////////////////////////////////////////////////////

    struct BMPINFOHEADER
    {
        size_t biSize;
        long biWidth;
        long biHeight;
        unsigned short biPlanes;
        unsigned short biBitCount;
        size_t biCompression;
        size_t biSizeImage;
        long biXPelsPerMeter;
        long biYPelsPerMeter;
        size_t biClrUsed;
        size_t biClrImportant;
    };
    //////////////////////////////////////////////////////////////////////////

    struct RGB4
    {
        byte B;
        byte G;
        byte R;
        byte A;
    };
    //////////////////////////////////////////////////////////////////////////

    struct RGB3
    {
        byte B;
        byte G;
        byte R;
    };
    //////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
       #pragma pack(pop)
#endif

    ifstream m_file;

    string m_sHdrBuf;
    BMPFILEHEADER *m_pFileHdr;
    BMPINFOHEADER *m_pBmpHdr;
    int m_cbOdd;
    bool m_bGrayScale;

    void checkResizeValid(const cudaExtent &e)
    {
        if(!this->m_pBuf)
        {
            if(e.depth == 3 || e.depth == 1)
            {
                this->m_dim = e;
            }
            else
                throw string("Can only assign X*Y and X*Y*3 buffers to a bitmap");
        }
    }
    //////////////////////////////////////////////////////////////////////////
    
    void initHeader()
    {
        size_t iFileHdrSize = sizeof(BMPFILEHEADER);
        size_t iBmpHdrSize = sizeof(BMPINFOHEADER);
        m_sHdrBuf.resize(iFileHdrSize + iBmpHdrSize);

        m_pBmpHdr = (BMPINFOHEADER*)&m_sHdrBuf[iFileHdrSize];
        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];

        BMPFILEHEADER &bfh = *m_pFileHdr;
        BMPINFOHEADER &bih = *m_pBmpHdr;

        int nPitch = (this->m_dim.width + 3) & ~3;
        bih.biBitCount = 24;
        bih.biClrImportant = 0;
        bih.biClrUsed = 0;
        bih.biCompression = 0;
        bih.biWidth = this->m_dim.width;
        bih.biHeight = this->m_dim.height;
        bih.biPlanes = 0;
        bih.biSize = iBmpHdrSize;
        bih.biSizeImage = nPitch * this->m_dim.height * (24 / 8);
        bih.biXPelsPerMeter = 2384; // 72 DPI
        bih.biYPelsPerMeter = 2384;

        bfh.bfReserved1 = 0;
        bfh.bfReserved2 = 0;
        bfh.bfSize = iFileHdrSize + iBmpHdrSize + bih.biSizeImage;
        bfh.bfType = 0x4d42;
        bfh.bfOffBits = iFileHdrSize + iBmpHdrSize;

        int nMod = (this->m_dim.width * (24 / 8)) % 4;
        m_cbOdd = nMod ? 4 - nMod : 0;
    }
    //////////////////////////////////////////////////////////////////////////
    
public:

    explicit TBitmap(bool bGrayScale = false) : m_bGrayScale(bGrayScale)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    TBitmap(int w, int h, bool bGrayScale = false) : m_bGrayScale(bGrayScale)
    {
        cudaExtent e;
        e.width = w;
        e.height = h;
        e.depth = m_bGrayScale ? 1 :3;
        checkResizeValid(e);
        this->resize();
        initHeader();
    }
    //////////////////////////////////////////////////////////////////////////

    void load(pcchar pszFile, bool bGrayScale = false)
    {

        m_bGrayScale = bGrayScale;
        m_file.open(pszFile, ios::binary);
        if(!m_file.is_open())
            throw string("Cannot open file : ") + pszFile;

        // Load BMP file header
        size_t iFileHdrSize = sizeof(BMPFILEHEADER);
        m_sHdrBuf.resize(iFileHdrSize);
        m_file.read(&m_sHdrBuf[0], iFileHdrSize);
        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];
        m_sHdrBuf.resize(m_pFileHdr->bfOffBits);

        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];
        m_file.read(&m_sHdrBuf[iFileHdrSize], m_pFileHdr->bfOffBits - iFileHdrSize);

        m_pBmpHdr = (BMPINFOHEADER*)&m_sHdrBuf[iFileHdrSize];

        if(m_pBmpHdr->biBitCount != 32 && m_pBmpHdr->biBitCount != 24)
            throw string("File format can only be 32 bpp or 24 bpp");

        this->m_dim.width = m_pBmpHdr->biWidth;
        this->m_dim.height = m_pBmpHdr->biHeight;
        this->m_dim.depth = bGrayScale ? 1 : 3;
        this->resize();

        // Calculate odd bytes for stride
        int cbPix = m_pBmpHdr->biBitCount == 24 ? 3 : 4;
        int nMod = (this->m_dim.width * cbPix) % 4;
        m_cbOdd = nMod ? 4 - nMod : 0;

        int w = this->width(), h = this->height();
        if(bGrayScale)
        {
            T3DView<T> p(this->data());
            for(int y = h - 1; y >= 0; --y)
            {
                for(int x = 0 ; x < w; ++x)
                {
                    RGB4 pix;
                    m_file.read((char*)&pix, cbPix);

                    p[0][y][x] = (T)
                        ((float)pix.R * .299 + 
                        (float)pix.G * .587 + 
                        (float)pix.B * .114);

                    //p[0][y][x] = (T)(((float)pix.R + (float)pix.G + (float)pix.B ) / 3);
                }

                // Skip odd data
                m_file.seekg(m_cbOdd, ios_base::cur);
            }
        }
        else
        {
            T3DView<T> p(this->data());
            for(int y = h - 1; y >= 0; --y)
            {
                for(int x = 0 ; x < w; ++x)
                {
                    RGB4 pix;
                    m_file.read((char*)&pix, cbPix);

                    p[0][y][x] = (T)pix.R;
                    p[1][y][x] = (T)pix.G;
                    p[2][y][x] = (T)pix.B;
                }

                // Skip odd data
                m_file.seekg(m_cbOdd, ios_base::cur);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////

    void save(pcchar pszFile, bool bGrayScale = false)
    {
        ofstream ofs(pszFile, ios::binary | ios::trunc);

        // copy the bitmap header data over directly
        ofs.write(m_sHdrBuf.c_str(), m_sHdrBuf.size());

        size_t cbPix = m_pBmpHdr->biBitCount == 24 ? 3 : 4;
        char dummy[4] = {0};

        if(bGrayScale)
        {
            T3DView<T> p(this->data());
            for(int y = this->height() - 1; y >= 0; --y)
            {
                for(int x = 0 ; x < this->width(); ++x)
                {
                    RGB4 pix;
                    float fVal = p[0][y][x];
                    pix.R = pix.G = pix.B = (byte)fVal;
                    pix.A = 255;
                    ofs.write((char*)&pix, cbPix);
                }

                // Write odd bytes
                ofs.write(dummy, m_cbOdd);
            }
        }
        else
        {
            T3DView<T> p(this->data());

            if(this->depth() == 1)
            {
                for(int y = this->height() - 1; y >= 0; --y)
                {
                    FOR(x, this->width())
                    {
                        RGB4 pix;
                        pix.R = (byte)p[0][y][x];
                        pix.G = (byte)p[0][y][x];
                        pix.B = (byte)p[0][y][x];
                        pix.A = 255;
                        ofs.write((char*)&pix, cbPix);
                    }
                }
            }
            else if(this->depth() == 3)
            {
                for(int y = this->height() - 1; y >= 0; --y)
                {
                    FOR(x, this->width())
                    {
                        RGB4 pix;
                        pix.R = (byte)p[0][y][x];
                        pix.G = (byte)p[1][y][x];
                        pix.B = (byte)p[2][y][x];
                        pix.A = 0;
                        ofs.write((char*)&pix, cbPix);
                    }

                    // Write odd bytes
                    ofs.write(dummy, m_cbOdd);
                }
            }
            else
                throw string("Unknown bitmap depth");
        }
    }
    //////////////////////////////////////////////////////////////////////////

    // Override default copy ctor
    explicit TBitmap(const TBitmap &that)
    {
        this->assign(&that);
        initHeader();
    }
    //////////////////////////////////////////////////////////////////////////

    explicit TBitmap(const TArray<T> &that)
    {
        checkResizeValid(that.dims());
        this->assign(&that);
        initHeader();
    }
    //////////////////////////////////////////////////////////////////////////

    // Override default = operator
    void operator=(const TBitmap &that)
    {
        this->assign(&that);
        initHeader();
    }
    //////////////////////////////////////////////////////////////////////////

    // Assignment from any type of buffer
    void operator=(const TArray<T> &that)
    {
        checkResizeValid(that.dims());
        this->assign(&that);
        initHeader();
    }
    //////////////////////////////////////////////////////////////////////////    

};
//////////////////////////////////////////////////////////////////////////

// makes a 1D a gaussian kernel in piResult of size (2 * iMid) + 1
template<typename T> void getGaussianKernel(int iMid, T *pResult) 
{
    float fSigma = iMid / 3.0F;
    float fCoEff = fSigma * fSigma * 2;

    for(int x = -iMid; x <= iMid; ++x)
    {
        int i = x + iMid;
        pResult[i] = iMid * exp(-x * x / fCoEff);
    }
}
//////////////////////////////////////////////////////////////////////////

void applyImageFilter(TArray<float> &pDataIn, TArray<float> &pDataOut, TArray<float> &pFilter);

void applyImageFilterFast(TArray<float> &pDataIn, TArray<float> &pDataOut, TArray<float> &pTemp, TArray<float> &pFilterSeparable);

#define getLinearThreadIndex(n) \
    int nBlock =                                                     \
    blockIdx.z * (gridDim.x * gridDim.y) +                       \
    blockIdx.y * (gridDim.x) +                                   \
    blockIdx.x;                                                  \
    n =                                                         \
    nBlock * (blockDim.x * blockDim.y * blockDim.z) +            \
    threadIdx.z * (blockDim.x * blockDim.y) +                    \
    threadIdx.y * (blockDim.x) +                                 \
    threadIdx.x;                                                 


//#define getThreadX(nMax) (getLinearThreadIndex() % nMax);

#define getThreadXY(pt, w, h)       \
    {                               \
    int nThread;                    \
    getLinearThreadIndex(nThread);  \
    pt.x = nThread % w;             \
    pt.y = (nThread /  w) % h;      \
    }

#define getThreadXYZ(pt, w, h, d)               \
{                                               \
    int nThread = getLinearThreadIndex();       \
    pt.x = nThread % w;                         \
    pt.y = (nThread / w) % h;                   \
    pt.z = (nThread / (w * h) ) % d;            \
}

#define clampval(val, mn, mx) { if(val < mn) val = mn; else if(val > mx) val = mx;}
//////////////////////////////////////////////////////////////////////////

#define getBlockGridSizes(nElems, nBlocks, nThreadsPerBlock)        \
{                                                                   \
    if((nElems) < g_devProps.multiProcessorCount * 8)               \
    {                                                               \
        nThreadsPerBlock = nElems;                                  \
        nBlocks = 1;                                                \
    }                                                               \
    else                                                            \
    {                                                               \
        nBlocks = g_devProps.multiProcessorCount;                   \
        nThreadsPerBlock = nElems / g_devProps.multiProcessorCount; \
                                                                    \
        if(nThreadsPerBlock > g_devProps.maxThreadsPerBlock)        \
        {                                                           \
            nThreadsPerBlock = g_devProps.maxThreadsPerBlock;       \
            nBlocks = nElems / nThreadsPerBlock;                    \
        }                                                           \
                                                                    \
        while((nThreadsPerBlock * nBlocks) < (nElems))              \
        {                                                           \
            if(nThreadsPerBlock == g_devProps.maxThreadsPerBlock)   \
                nBlocks++;                                          \
            else                                                    \
                nThreadsPerBlock++;                                 \
        }                                                           \
        if(nThreadsPerBlock * nBlocks > g_devProps.maxGridSize[0] * g_devProps.maxThreadsDim[0]) throw string("Too many threads for kernel to execute!");\
    }                                                               \
}                                                                   
//////////////////////////////////////////////////////////////////////////


#endif //__CUDALIB_H__

