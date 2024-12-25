#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <omp.h>  

void transpose_matrix(int** src, int** dest, const int ROW, const int COL)
{
    #pragma omp parallel for collapse(2)  
    for (int i = 0; i < ROW; ++i)
    {
        for (int j = 0; j < COL; ++j)
        {
            dest[j][i] = src[i][j]; 
        }
    }
}

void matrix_multiplication_with_transpose(int** A, int** B_T, int** C, const int ROW, const int COL, int startRow, int endRow)
{
    #pragma omp parallel for collapse(2)  
    for (int i = startRow; i < endRow; ++i)
    {
        for (int j = 0; j < COL; ++j)
        {
            C[i][j] = 0;
            for (int k = 0; k < ROW; ++k)
            {
                C[i][j] += A[i][k] * B_T[j][k]; 
            }
        }
    }
}

void generate_matrix(int** matrix, const size_t ROW, const size_t COL)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 10);
    
    #pragma omp parallel for collapse(2)  
    for (int i = 0; i < ROW; ++i)
    {
        for (int j = 0; j < COL; ++j)
        {
            matrix[i][j] = distr(gen);
        }
    }
}

void operation(int** A, int** B, int** C, int** B_T, const size_t ROW, const size_t COL)
{
    generate_matrix(A, ROW, COL);
    generate_matrix(B, ROW, COL);

    transpose_matrix(B, B_T, ROW, COL);

    auto t_start = std::chrono::high_resolution_clock::now();

    int numThreads = std::thread::hardware_concurrency();  
    int rowsPerThread = ROW / numThreads;


    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int startRow = tid * rowsPerThread;
        int endRow = (tid == numThreads - 1) ? ROW : (tid + 1) * rowsPerThread;

        matrix_multiplication_with_transpose(A, B_T, C, ROW, COL, startRow, endRow);
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "Processing time of " << ROW << " x " << COL << " matrix: " << elapsed_time_ms << " ms" << std::endl;
}

template <size_t ARRAY_SIZE>
void create_and_operate()
{
    int **A, **B, **C, **B_T;
    A = new int*[ARRAY_SIZE];
    B = new int*[ARRAY_SIZE];
    C = new int*[ARRAY_SIZE];
    B_T = new int*[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        A[i] = new int[ARRAY_SIZE];
        B[i] = new int[ARRAY_SIZE];
        C[i] = new int[ARRAY_SIZE];
        B_T[i] = new int[ARRAY_SIZE];
    }

    operation(A, B, C, B_T, ARRAY_SIZE, ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
        delete[] B_T[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] B_T;
}

int main()
{
    std::cout << "size: " << 3 << std::endl;
    create_and_operate<3>();
    std::cout << "size: " << 8 << std::endl;
    create_and_operate<8>();
    std::cout << "size: " << 16 << std::endl;
    create_and_operate<16>();
    std::cout << "size: " << 32 << std::endl;
    create_and_operate<32>();
    std::cout << "size: " << 64 << std::endl;
    create_and_operate<64>();
    std::cout << "size: " << 128 << std::endl;
    create_and_operate<128>();
    std::cout << "size: " << 256 << std::endl;
    create_and_operate<256>();
    std::cout << "size: " << 512 << std::endl;
    create_and_operate<512>();
    std::cout << "size: " << 1024 << std::endl;
    create_and_operate<1024>();
    std::cout << "size: " << 2048 << std::endl;
    create_and_operate<2048>();
    std::cout << "size: " << 4096 << std::endl;
    create_and_operate<4096>();
    std::cout << "size: " << 8192 << std::endl;
    create_and_operate<8192>();
    std::cout << "size: " << 16384 << std::endl;
    create_and_operate<16384>();
    return 0;
}
