#include <iostream>
#include <armadillo>
using namespace std;

int main() {
    arma::mat matrix;
    arma::mat matrix_ = {
        {1, 2, 3},
        {4, 5, 6}
    };
    arma::vec v = {1,2 ,2};

    // matrix = matrix_;
    matrix = v;
    matrix.print("this is the matrix");

    return 0;
}
