#include "Laplacian.h"
#include "Parameters.h"
#include "PointwiseOps.h"
#include "Reductions.h"
#include "Utilities.h"
#include "Timer.h"

#include <iostream>

extern Timer timerLaplacian;
extern Timer timerSaxpyLine2;
extern Timer timerSaxpyLine8;
extern Timer timerSaxpyLine9;
extern Timer timerSaxpyLine16_1;
extern Timer timerSaxpyLine16_2;
extern Timer timerCopyLine4;
extern Timer timerCopyLine13;
extern Timer innerProductLine4;
extern Timer innerProductLine6;
extern Timer innerProductLine13;
extern Timer timerNormLine2;
extern Timer timerNormLine8;

void ConjugateGradients(
    float (&x)[XDIM][YDIM][ZDIM],
    const float (&f)[XDIM][YDIM][ZDIM],
    float (&p)[XDIM][YDIM][ZDIM],
    float (&r)[XDIM][YDIM][ZDIM],
    float (&z)[XDIM][YDIM][ZDIM],
    const bool writeIterations)
{
    // Algorithm : Line 2
    timerLaplacian.Restart();
    ComputeLaplacian(x, z);
    timerLaplacian.Pause();

    timerSaxpyLine2.Restart();
    Saxpy(z, f, r, -1);
    timerSaxpyLine2.Pause();
    timerNormLine2.Restart();
    float nu = Norm(r);
    timerNormLine2.Pause();

    // Algorithm : Line 3
    if (nu < nuMax)
        return;

    // Algorithm : Line 4
    timerCopyLine4.Restart();
    Copy(r, p);
    timerCopyLine4.Pause();
    innerProductLine4.Restart();
    float rho = InnerProduct(p, r);
    innerProductLine4.Pause();

    // Beginning of loop from Line 5
    for (int k = 0;; k++)
    {
        std::cout << "Residual norm (nu) after " << k << " iterations = " << nu << std::endl;

        // Algorithm : Line 6
        timerLaplacian.Restart();
        ComputeLaplacian(p, z);
        timerLaplacian.Pause();
        innerProductLine6.Restart();
        float sigma = InnerProduct(p, z);
        innerProductLine6.Pause();

        // Algorithm : Line 7
        float alpha = rho / sigma;

        // Algorithm : Line 8
        timerSaxpyLine8.Restart();
        Saxpy(z, r, r, -alpha);
        timerSaxpyLine8.Pause();
        timerNormLine8.Restart();
        nu = Norm(r);
        timerNormLine8.Pause();

        // Algorithm : Lines 9-12
        if (nu < nuMax || k == kMax)
        {
            timerSaxpyLine9.Restart();
            Saxpy(p, x, x, alpha);
            timerSaxpyLine9.Pause();
            std::cout << "Conjugate Gradients terminated after " << k << " iterations; residual norm (nu) = " << nu << std::endl;
            if (writeIterations)
                WriteAsImage("x", x, k, 0, 127);
            return;
        }

        // Algorithm : Line 13
        timerCopyLine13.Restart();
        Copy(r, z);
        timerCopyLine13.Pause();
        innerProductLine13.Restart();
        float rho_new = InnerProduct(z, r);
        innerProductLine13.Pause();

        // Algorithm : Line 14
        float beta = rho_new / rho;

        // Algorithm : Line 15
        rho = rho_new;

        // Algorithm : Line 16
        timerSaxpyLine16_1.Restart();
        Saxpy(p, x, x, alpha);
        timerSaxpyLine16_1.Pause();
        timerSaxpyLine16_2.Restart();
        Saxpy(p, r, p, beta);
        timerSaxpyLine16_2.Pause();

        if (writeIterations)
            WriteAsImage("x", x, k, 0, 127);
    }
}
