#pragma once
#include <Eigen/Core>
#include "load_vector.hpp"

//----------------AssembleVectorBegin----------------
//! Assemble the load vector into the full right hand side
//! for the linear system
//!
//! @param[out] F will at the end contain the RHS values for each vertex.
//! @param[in] vertices a list of triangle vertices
//! @param[in] triangles a list of triangles
//! @param[in] f the RHS function f.
void assembleLoadVector(Eigen::VectorXd &F,
                        const Eigen::MatrixXd &vertices,
                        const Eigen::MatrixXi &triangles,
                        const std::function<double(double, double)> &f)
{
     const int numberOfElements = triangles.rows();

     F.resize(vertices.rows());
     F.setZero();
     // (write your solution here)
     for (int i = 0; i < numberOfElements; i += 3)
     {
          const auto &a = vertices.row(triangles.row(i)(0));
          const auto &b = vertices.row(triangles.row(i)(1));
          const auto &c = vertices.row(triangles.row(i)(2));
          Eigen::Vector3d vector_element;
          computeLoadVector(vector_element, a, b, c, f);
          F(triangles.row(i)(0)) = vector_element(0);
          F(triangles.row(i)(1)) = vector_element(1);
          F(triangles.row(i)(2)) = vector_element(2);
     }
}
//----------------AssembleVectorEnd----------------
