#include "MatMatMultiply.h"
#include "Timer.h"
#include "Utilities.h"

#include <iostream>
#include <iomanip>

int main(int argc, char *argv[])
{
    float *Araw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64 ) );
    float *Braw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64 ) );
    float *Craw = static_cast<float*>( AlignedAllocate( MATRIX_SIZE * MATRIX_SIZE * sizeof(float), 64 ) );

    using matrix_t = float (&) [MATRIX_SIZE][MATRIX_SIZE];

    matrix_t A = reinterpret_cast<matrix_t>(*Araw);
    matrix_t B = reinterpret_cast<matrix_t>(*Braw);
    matrix_t C = reinterpret_cast<matrix_t>(*Craw);

    InitializeMatrices(A, B);

    Timer timer;

    int num_test = 20;
    double avg_time = 0;

    for(int test = 1; test <= num_test; test++)
    {
        std::cout << "Running test iteration " << std::setw(2) << test << " ";
        timer.Start();
        MatMatMultiply(A, B, C);
        avg_time += timer.Stop("Elapsed time : ");
    }

    std::cout << "\nMatrix Size: " << MATRIX_SIZE << std::endl;
    std::cout << "Average Time: " << avg_time / num_test << "ms" << std::endl;
    
    return 0;
}
