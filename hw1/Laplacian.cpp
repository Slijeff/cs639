#include "Laplacian.h"

void ComputeLaplacian(const float (&u)[XDIM][YDIM][ZDIM],
                      float (&Lu)[XDIM][YDIM][ZDIM])
{

#pragma omp parallel for
  for (int i = 1; i < XDIM - 1; i++)
    for (int k = 1; k < YDIM - 1; k++)
      for (int j = 1; j < ZDIM - 1; j++)
        Lu[i][j][k] = -6 * u[i][j][k] + u[i + 1][j][k] + u[i - 1][j][k] +
                      u[i][j + 1][k] + u[i][j - 1][k] + u[i][j][k + 1] +
                      u[i][j][k - 1];
}
