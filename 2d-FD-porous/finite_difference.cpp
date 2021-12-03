#include <Eigen/Sparse>
#include <iostream>
#include "writer.hpp"
#include <cmath>
#include <Eigen/SparseCholesky>
#include <stdexcept>
#include <iostream>

//! Sparse Matrix type. Makes using this type easier.
typedef Eigen::SparseMatrix<double> SparseMatrix;

//! Used for filling the sparse matrix.
typedef Eigen::Triplet<double> Triplet;

//! Vector type
typedef Eigen::VectorXd Vector;

//! Our function pointer, typedef'd to make it easier to use
typedef double (*FunctionPointer)(double, double);

//----------------poissonBegin----------------
//! Create the Poisson matrix for 2D finite difference.
//! @param[out] A will be the Poisson matrix (as in the exercise)
//! @param[in] N number of elements in the x-direction
void createPorousMediaMatrix2D(SparseMatrix &A, FunctionPointer sigma, int N, double dx)
{
    std::vector<Triplet> triplets;
    A.resize(N * N, N * N);
    triplets.reserve(5 * N * N - 4 * N);

    // Fill up triples
    std::cout << "calculating A matrix..." << std::endl;

    auto x = [&](int i)
    { return i * dx; };
    auto y = [&](int j)
    { return j * dx; };

    for (int j = 1; j <= N; ++j)
    {
        for (int i = 1; i <= N; ++i)
        {
            triplets.push_back(Triplet(N * (j - 1) + i - 1, N * (j - 1) + i - 1, sigma(x(i + 0.5), y(j)) + sigma(x(i - 0.5), y(j)) + sigma(x(i), y(j + 0.5)) + sigma(x(i), y(j - 0.5))));
            if (i < N)
            {
                //right of the diagonal entry
                triplets.push_back(Triplet(N * (j - 1) + i, N * (j - 1) + i - 1, -sigma(x(i + 0.5), y(j))));
                //below the diagonal entry
                triplets.push_back(Triplet(N * (j - 1) + i - 1, N * (j - 1) + i, -sigma(x(i + 0.5), y(j))));
            }
            if (j < N)
            {
                //right of B
                triplets.push_back(Triplet(N * j + i - 1, N * (j - 1) + i - 1, -sigma(x(i), y(j + 0.5))));
                //below B
                triplets.push_back(Triplet(N * (j - 1) + i - 1, N * j + i - 1, -sigma(x(i), y(j + 0.5))));
            }
        }
    }

    // (write your solution here)
    A.setFromTriplets(triplets.begin(), triplets.end());
}

//----------------poissonEnd----------------

//----------------RHSBegin----------------
//! Create the Right hand side for the 2D finite difference
//! @param[out] rhs will at the end contain the right hand side
//! @param[in] f the right hand side function f
//! @param[in] N the number of points in the x direction
//! @param[in] dx the cell width
void createRHS(Vector &rhs, FunctionPointer f, int N, double dx)
{
    std::cout << "calculating RHS..." << std::endl;
    rhs.resize(N * N);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            const double x = (i + 1) * dx;
            const double y = (j + 1) * dx;
            rhs[j * N + i] = dx * dx * f(x, y);
        }
    }
}
//----------------RHSEnd----------------

//----------------solveBegin----------------
//! Solve the Poisson equation in 2D
//! @param[in] f the function pointer to f
//! @param[in] N the number of points to use (in x direction)
//!
//! @returns the approximate solution u
Vector poissonSolve(FunctionPointer f, FunctionPointer sigma, int N)
{
    Vector u;
    double dx = 1.0 / (N + 1);
    std::cout << "solving..." << std::endl;
    // Compute A, rhs and u
    // (write your solution here)
    SparseMatrix A;
    Vector rhs;
    createPorousMediaMatrix2D(A, sigma, N, dx);
    createRHS(rhs, f, N, dx);
    //solve the system
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Decomposition failed");
    }

    u = solver.solve(rhs);
    Vector solution((N + 2) * (N + 2));
    solution.setZero();
    for (int j = 1; j <= N; ++j)
    {
        for (int i = 1; i <= N; ++i)
        {
            solution[j * (N + 2) + i] = u((j - 1) * N + (i - 1));
        }
    }
    return solution;
}
//----------------solveEnd----------------

//! Gives the Right hand side F at the point x, y
double F(double x, double y)
{
    return 4 * M_PI * M_PI * sin(2 * M_PI * x) * sin(2 * M_PI * y) * (4 * cos(2 * M_PI * x) * cos(2 * M_PI * y) + M_PI);
}

//----------------convergenceBegin----------------
//! Gives the exact solution at the point x,y
double exactSolution(double x, double y)
{
    // (write your solution here)
    return sin(2 * M_PI * x) * sin(2 * M_PI * y);
    //return 0; //remove when implemented
}

void convergeStudy(FunctionPointer F, FunctionPointer sigma)
{
    const int startK = 3;
    const int endK = 9;

    Vector errors(endK - startK);
    Vector resolutions(errors.size());
    for (int k = startK; k < endK; ++k)
    {
        const int N = 1 << k; // 2^k
        std::cout << "calculating error for N = " << N << std::endl;
        // (write your solution here)
        int i = k - startK;
        resolutions[i] = N;
        Vector diff(N * N);
        double dx = 1.0 / (N + 1);
        auto approx = poissonSolve(F, sigma, N);
        for (int j = 1; j <= N; ++j)
        {
            for (int i = 1; i <= N; ++i)
            {
                diff[(j - 1) * N + i - 1] = approx[j * (N + 2) + i] - exactSolution(i * dx, j * dx);
            }
        }
        errors[i] = diff.lpNorm<Eigen::Infinity>();
    }

    writeToFile("errors.txt", errors);
    writeToFile("resolutions.txt", resolutions);
}
//----------------convergenceEnd----------------

int main(int, char **)
{
    auto sigmaCos = [](double x, double y)
    {
        return M_PI / 2. + cos(2 * M_PI * x) * cos(2 * M_PI * y);
    };

    auto sigmaConstant = [](double x, double y)
    {
        return 1.0;
    };
    auto u = poissonSolve(F, sigmaCos, 500);
    std::cout << "write to file" << std::endl;
    writeToFile("u_fd.txt", u);

    convergeStudy(F, sigmaCos);
}
