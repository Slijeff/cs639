#include "Timer.h"
#include "Laplacian.h"

#include <iomanip>
#include <omp.h>

int main(int argc, char *argv[])
{
    using array_t = float(&)[XDIM][YDIM][ZDIM];

    float *uRaw = new float[XDIM * YDIM * ZDIM];
    float *LuRaw = new float[XDIM * YDIM * ZDIM];
    array_t u = reinterpret_cast<array_t>(*uRaw);
    array_t Lu = reinterpret_cast<array_t>(*LuRaw);

    Timer timer;
    omp_set_num_threads(6);
    for (int test = 1; test <= 10; test++)
    {
        std::cout << "Running test iteration " << std::setw(2) << test << " ";
        timer.Start();
        ComputeLaplacian(u, Lu);
        timer.Stop("Elapsed time : ");
    }

    return 0;
}
