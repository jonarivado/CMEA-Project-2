#include "create_poisson_matrix.hpp"

//! Used for filling the sparse matrix.
using Triplet = Eigen::Triplet<double>;

//! Create the 1D Poisson matrix
//! @param[in] N the number of interior points
//! @param[in] a the coefficient function a
//!
//! @returns the Poisson matrix.
SparseMatrix createPoissonMatrix(int N,
                                 const std::function<double(double)> &a)
{

    SparseMatrix A;
    // (write your solution here)
    A.resize(N, N);
    std::vector<Triplet> triplets;

    double dx = 1.0 / (N + 1);
    for (int i = 0; i < N; i++)
    {
        double x = (i + 1) * dx;
        double a_x = a(x);
        triplets.push_back(Triplet(i, i, 2 * a_x / (dx * dx)));
        if (i > 0)
        {
            triplets.push_back(Triplet(i, i - 1, -1 * (a_x) / (dx * dx)));
        }
        if (i < N - 1)
        {
            triplets.push_back(Triplet(i, i + 1, -1 * (a_x) / (dx * dx)));
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}
