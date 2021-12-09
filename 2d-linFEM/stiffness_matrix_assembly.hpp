#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include "stiffness_matrix.hpp"

//! Sparse Matrix type. Makes using this type easier.
typedef Eigen::SparseMatrix<double> SparseMatrix;

//! Used for filling the sparse matrix.
typedef Eigen::Triplet<double> Triplet;

//----------------AssembleMatrixBegin----------------
//! Assemble the stiffness matrix
//! for the linear system
//!
//! @param[out] A will at the end contain the Galerkin matrix
//! @param[in] vertices a list of triangle vertices
//! @param[in] triangles a list of triangles
//! @param[in] sigma the function sigma as in the exercise
//! @param[in] r the parameter r as in the exercise
template <class Matrix>
void assembleStiffnessMatrix(Matrix &A, const Eigen::MatrixXd &vertices,
                             const Eigen::MatrixXi &triangles,
                             const std::function<double(double, double)> &sigma = constantFunction,
                             double r = 0)
{

    const int numberOfElements = triangles.rows();
    A.resize(vertices.rows(), vertices.rows());

    std::vector<Triplet> triplets;

    triplets.reserve(numberOfElements * 3 * 3);
    // (write your solution here)
    for (int i = 0; i < numberOfElements; i++)
    {
        /*const auto &a = vertices(triangles(i, 0), 0);
        const auto &b = vertices(triangles(i, 1), 1);
        const auto &c = vertices(triangles(i, 2), 2);*/
        const auto &a = vertices.row(triangles.row(i)(0));
        const auto &b = vertices.row(triangles.row(i)(1));
        const auto &c = vertices.row(triangles.row(i)(2));

        Eigen::Matrix3d A_element;
        computeStiffnessMatrix(A_element, a, b, c, sigma, r);

        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                triplets.push_back(Triplet(triangles.row(i)(j), triangles.row(i)(k), A_element(j, k)));
            }
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
}
//----------------AssembleMatrixEnd----------------
