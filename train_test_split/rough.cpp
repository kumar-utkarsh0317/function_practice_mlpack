#include <iostream>
using namespace std;
#include <armadillo>
#include"utkarsh.hpp"
using namespace utkarsh;

int main(){
    // arma::uvec v = arma::linspace<arma::uvec>(0, 100 - 1, 100);
    // v = arma::shuffle(v);
    arma::mat A = arma::randn<arma::mat>(5, 5);
    A.print();
        // Slice a submatrix (e.g., rows 1 to 3 and columns 2 to 4)
    arma::mat submatrix = A.submat(1, 2, 3, 4);

    // Print the sliced submatrix
    std::cout << "Sliced Submatrix:\n" << submatrix << "\n";

    // v.print();

    return 0;
}