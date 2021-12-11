#include "crank_nicolson.hpp"

//! Uses the Crank-Nicolson method to compute u from time 0 to time T
//!
//! @param[in] u0 the initial data, as column vector (size N+2)
//! @param[in] dt the time step size
//! @param[in] T the final time at which to compute the solution (which we assume to be a multiple of dt)
//! @param[in] N the number of interior grid points
//! @param[in] gL function of time with the Dirichlet condition at left boundary
//! @param[in] gR function of time with the Dirichlet condition at right boundary
//! @param[in] a the coefficient function a
//!
//! @note the vector returned should include the boundary values!
//!
std::pair<Eigen::MatrixXd, Eigen::VectorXd> crankNicolson(
    const Eigen::VectorXd &u0,
    double dt, double T, int N,
    const std::function<double(double)> &gL,
    const std::function<double(double)> &gR,
    const std::function<double(double)> &a)
{

    Eigen::VectorXd time;
    Eigen::MatrixXd u;

    // (write your solution here)
    const int nsteps = int(round(T / dt));
    const double h = 1.0 / (N + 1);
    u.resize(N + 2, nsteps + 1);
    time.resize(nsteps + 1);

    // Initial conditions
    u.col(0) = u0;
    time(0) = 0;

    auto A = createPoissonMatrix(N, a);
    SparseMatrix I(N, N);
    I.setIdentity();
    auto LHS = I + dt / 2 * A;
    auto RHS = I - dt / 2 * A;
    Eigen::SparseLU<SparseMatrix> solver;
    solver.compute(LHS);
    for (int t = 1; t <= nsteps; t++)
    {
        u(0, t) = dt * gL(t * dt) + u.col(t - 1)(0) - dt * a(h) * u.col(t - 1)(0) / (h * h);
        u(N + 1, t) = dt * gR(t * dt) + u.col(t - 1)(N) - dt * a(1 - h) * u.col(t - 1)(N) / (h * h);
    }
    for (int t = 1; t <= nsteps; t++)
    {
        time(t) = t * dt;
        u.col(t).segment(1, N) = solver.solve(RHS * u.col(t - 1).segment(1, N));
    }
    return std::make_pair(u, time);
}
