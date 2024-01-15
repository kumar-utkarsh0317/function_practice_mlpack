#include <mlpack.hpp>
#include<armadillo>
using namespace std;

int main()
{
        // Create a 3x3x2 cube with custom weights.
        arma::Cube<double> original_cube(3, 3, 2);

        // Initialize the cube with custom weights.
        original_cube.slice(0) = {{1.0, 2.0, 3.0},
                                {4.0, 5.0, 6.0},
                                {7.0, 8.0, 9.0}};

        original_cube.slice(1) = {{10.0, 11.0, 12.0},
                                {13.0, 14.0, 15.0},
                                {16.0, 17.0, 18.0}};

        original_cube.print("This is the original cube");

        // Reshape the cube to a matrix before printing.
         arma::Mat<double> reshaped_matrix = arma::reshape(original_cube, 9, 2, 1);
        // original_cube.reshape(9, 2, 1);
        // original_cube.print("new matrix");
        reshaped_matrix.print("reshaped one");




    return 0;
}
