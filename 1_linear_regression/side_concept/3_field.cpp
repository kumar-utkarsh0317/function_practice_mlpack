#include <iostream>
#include <armadillo>

using namespace arma;

int main() {
    mat A = randn(2, 3);
    mat B = randn(4, 5);

    // Correct instantiation of field without parentheses
    field<mat> F;

    F(0) = A;
    F(1) = B;

    F.print("F:");

    return 0;
}
