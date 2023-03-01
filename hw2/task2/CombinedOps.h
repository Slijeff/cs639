#pragma once

#include "Parameters.h"

void CombinedSaxpyLine16(float (&x)[XDIM][YDIM][ZDIM], float (&p)[XDIM][YDIM][ZDIM],
                         const float (&z)[XDIM][YDIM][ZDIM], const float alpha, const float beta);

float CombinedLine6(const float (&u)[XDIM][YDIM][ZDIM], float (&Lu)[XDIM][YDIM][ZDIM]);