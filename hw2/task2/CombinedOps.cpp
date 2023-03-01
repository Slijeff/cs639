#include "CombinedOps.h"

void CombinedSaxpyLine16(float (&x)[XDIM][YDIM][ZDIM], float (&p)[XDIM][YDIM][ZDIM],
                         const float (&z)[XDIM][YDIM][ZDIM], const float alpha, const float beta)
{
    for (int i = 1; i < XDIM - 1; i++)
        for (int j = 1; j < YDIM - 1; j++)
            for (int k = 1; k < ZDIM - 1; k++)
            {
                x[i][j][k] = p[i][j][k] * alpha + x[i][j][k];
                p[i][j][k] = p[i][j][k] * beta + z[i][j][k];
            };
}

float CombinedLine6(const float (&x)[XDIM][YDIM][ZDIM], float (&y)[XDIM][YDIM][ZDIM])
{
    double result = 0.;
#pragma omp parallel for reduction(+ \
                                   : result)
    for (int i = 1; i < XDIM - 1; i++)
        for (int j = 1; j < YDIM - 1; j++)
            for (int k = 1; k < ZDIM - 1; k++)
            {
                y[i][j][k] =
                    -6 * x[i][j][k] + x[i + 1][j][k] + x[i - 1][j][k] + x[i][j + 1][k] + x[i][j - 1][k] + x[i][j][k + 1] + x[i][j][k - 1];
                result += (double)x[i][j][k] * (double)y[i][j][k];
            }
    return (float)result;
}
