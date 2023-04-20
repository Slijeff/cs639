#include "Reductions.h"

#include <algorithm>
#include <mkl.h>

float Norm(const float (&x)[XDIM][YDIM][ZDIM])
{
    float result = 0.;

    int index = cblas_isamax (XDIM * YDIM * ZDIM, &x[0][0][0], 1);
    result = x[index / (YDIM * ZDIM)][(index / ZDIM) % YDIM][index % ZDIM];

    return std::abs(result);
}

float InnerProduct(const float (&x)[XDIM][YDIM][ZDIM], const float (&y)[XDIM][YDIM][ZDIM])
{
    float result = cblas_sdot(XDIM * YDIM * ZDIM, &x[0][0][0], 1, &y[0][0][0], 1);

    return result;
}
