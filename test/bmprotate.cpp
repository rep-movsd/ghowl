#include "ghowl.h"

// Get the integer co-ordinates that are nearest to a point
template<typename T>
GPU_CPU void getNearestPixels(T& pt, TMany<4, TXY> &bounds)
{
    bounds[0].x = (int)pt.x;
    bounds[0].y = (int)pt.y;

    bounds[1].x = (int)pt.x;
    bounds[1].y = (int)pt.y + 1;

    bounds[2].x = (int)pt.x + 1;
    bounds[2].y = (int)pt.y + 1;

    bounds[3].x = (int)pt.x + 1;
    bounds[3].y = (int)pt.y;
}
///////////////////////////////////////////////////////////////////

// Interpolate the value at fDist if the value at 0 is f1 and at 1 is f2
GPU_CPU float interpolate(float fDist, float f1, float f2)
{
    return f1 + ((f2 - f1) * fDist);
}
//////////////////////////////////////////////////////////////////////////

KERNEL void fnRotate(TPitchPtr ps, TPitchPtr pd, float fTheta)
{
    // Which pixel is this thread operating on?
    TXY pt;
    getThreadXY(pt, ps.w, ps.h);

    // Get the midpoint of the image
    TFXY ptMid(ps.w, ps.h);
    ptMid /= 2;

    // Get coordinates relative to image center and rotate the point to get the co-ord of the source
    TFXY ptSrc = TFXY(pt) - ptMid;
    ptSrc.rotate(fTheta);

    // translate back to normal co ords
    ptSrc += ptMid;

    // Get the four integer co-ords nearest to that point
    // They are in clockwise starting from bottom left
    TMany<4, TXY> p;
    getNearestPixels(ptSrc, p);

    // for each plane R G and B
    FOR(z, 3)
    {
        // Create a view to write the destination pixel (choose the z-th plane from the 3 planes in the bitmap)
        T2DView<byte> d(pd, z);

        // Since the point we are interpolating from could be outside the bounds of the image, we use a default 2d view
        // Access outside the specified range will get a value 0 (defined by the last but one parameter)
        T2DDefView<byte> s(ps, 1, ps.w - 1, 1, ps.h - 1, 0, z);

        float dx = ptSrc.x - p[0].x;
        // First interpolate horizontally between p0 and p3 and then between p1 and p2

        float fi1 = interpolate(dx, s[p[0]], s[p[3]]);
        float fi2 = interpolate(dx, s[p[1]], s[p[2]]);

        // next interpolate vertically between fi1 and fi2
        d[pt] = (byte)interpolate(ptSrc.y - p[0].y, fi1, fi2);
    }
}
//////////////////////////////////////////////////////////////////////////

void rotateBMP(char *szInfile, char *szOutFile, float fTheta)
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
        KERNEL_1D_GRID_1D(fnRotate, nBlock, nThreadsPerBlock)
            (src.data(), dst.data(), fTheta);

        // Assign device memory back to bitmap and save it to output file
        bmp = dst;
        bmp.save(szOutFile);
    }
    catch(TCudaException &e)
    {
        throw e.message();
    }
}
