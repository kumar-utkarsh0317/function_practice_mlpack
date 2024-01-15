// #include <mlpack.hpp>
#include <armadillo>

int main() {
//     arma::mat matrix;
//     std::string path = "/home/kumarutkarsh/Desktop/mlpack_function_practice/5_convolution_output/side_concept/rough.csv";
//     mlpack::data::Load(path, matrix);
//     matrix.print("matrix");

//     // Reshape the matrix in-place into a cube (assuming a desired number of slices)
//     int desired_slices = 1;  // Example: 3 slices
//     matrix.reshape(matrix.n_rows, matrix.n_cols, desired_slices);

//     // Now matrix is a cube with the specified dimensions
//     matrix.print("Reshaped cube:");




        // arma::mat matrix;
        // std::string path = "/home/kumarutkarsh/Desktop/mlpack_function_practice/5_convolution_output/side_concept/rough.csv";
        // mlpack::data::Load(path, matrix);
        // matrix.print("matrix");

        // // Reshape the matrix in-place into a cube (assuming a desired number of slices)
        // int desired_slices = 1;  // Example: 3 slices
        // matrix.reshape(6, 2);

        // // Now matrix is a cube with the specified dimensions
        // matrix.print("Reshaped cube:");


        arma::mat matrix = {
                {1, 2},
                {3, 4}
        };

        // arma::cube c = 
        matrix.reshape(matrix.n_rows, matrix.n_cols, 1);
        matrix.print("this is the cue");

        return 0;
}

