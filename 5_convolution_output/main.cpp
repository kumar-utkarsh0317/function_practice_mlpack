#include<mlpack.hpp>
using namespace std;



int main()
{
        /*
        loading the image as matrix
        
        remember that the matrix is in 2d form having */
        // arma::mat dataset;
        // string csv_path = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/mnist_dataset/train.csv";
        // mlpack::data::Load(csv_path, mdataset);
        
        #row, col, channels
        arma::cube x(4, 5, 6, arma::fill::randu);
        x.print();

        arma::mat b = x.slice()

}