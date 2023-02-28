#include "ConjugateGradients.h"
#include "Timer.h"
#include "Utilities.h"
#include "omp.h"
#include "vector"

Timer timerLaplacian;
Timer timerSaxpyLine2;
Timer timerSaxpyLine8;
Timer timerSaxpyLine9;
Timer timerSaxpyLine16_1;
Timer timerSaxpyLine16_2;
Timer timerCopyLine4;
Timer timerCopyLine13;
Timer innerProductLine4;
Timer innerProductLine6;
Timer innerProductLine13;
Timer timerNormLine2;
Timer timerNormLine8;
Timer total;

int main(int argc, char *argv[])
{
    using array_t = float(&)[XDIM][YDIM][ZDIM];

    float *xRaw = new float[XDIM * YDIM * ZDIM];
    float *fRaw = new float[XDIM * YDIM * ZDIM];
    float *pRaw = new float[XDIM * YDIM * ZDIM];
    float *rRaw = new float[XDIM * YDIM * ZDIM];
    float *zRaw = new float[XDIM * YDIM * ZDIM];

    array_t x = reinterpret_cast<array_t>(*xRaw);
    array_t f = reinterpret_cast<array_t>(*fRaw);
    array_t p = reinterpret_cast<array_t>(*pRaw);
    array_t r = reinterpret_cast<array_t>(*rRaw);
    array_t z = reinterpret_cast<array_t>(*zRaw);

    omp_set_num_threads(6);

    // Initialization
    {
        Timer timer;
        timer.Start();
        InitializeProblem(x, f);
        timer.Stop("Initialization : ");
    }

    std::vector<Timer> timers = {
        timerLaplacian,
        timerSaxpyLine2,
        timerSaxpyLine8,
        timerSaxpyLine9,
        timerSaxpyLine16_1,
        timerSaxpyLine16_2,
        timerCopyLine4,
        timerCopyLine13,
        innerProductLine6,
        innerProductLine4,
        innerProductLine13,
        timerNormLine2,
        timerNormLine8,
        total};
    // Call Conjugate Gradients algorithm
    for (auto t : timers)
    {
        t.Reset();
    }
    total.Restart();
    ConjugateGradients(x, f, p, r, z, false);
    total.Pause();
    timerLaplacian.Print("Total Laplacian Time : ");
    timerSaxpyLine2.Print("Total Saxpy Time on Line 2: ");
    timerSaxpyLine8.Print("Total Saxpy Time on Line 8: ");
    timerSaxpyLine9.Print("Total Saxpy Time on Line 9: ");
    timerSaxpyLine16_1.Print("Total Saxpy Time on Line 16 (first): ");
    timerSaxpyLine16_2.Print("Total Saxpy Time on Line 16 (second): ");
    timerCopyLine4.Print("Total Copy Time on Line 4: ");
    timerCopyLine13.Print("Total Copy Time on Line 13: ");
    innerProductLine4.Print("Total innerProduct Time on Line 4: ");
    innerProductLine6.Print("Total innerProduct Time on Line 6: ");
    innerProductLine13.Print("Total innerProduct Time on Line 13: ");
    timerNormLine2.Print("Total Norm Time on Line 2: ");
    timerNormLine8.Print("Total Norm Time on Line 8: ");
    total.Print("Total Conjugate Gradients time: ");

    return 0;
}
