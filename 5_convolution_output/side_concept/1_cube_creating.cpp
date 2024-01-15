#include<mlpack.hpp>
using namespace std;



int main()
{
        // Create a 3x3x2 cube with custom weights.
        arma::Cube<double> customCube(3, 3, 2);

        // Initialize the cube with custom weights.
        customCube.slice(0) = {{1.0, 2.0, 3.0},
                                {4.0, 5.0, 6.0},
                                {7.0, 8.0, 9.0}};

        customCube.slice(1) = {{10.0, 11.0, 12.0},
                                {13.0, 14.0, 15.0},
                                {16.0, 17.0, 18.0}};

        // Print the custom cube.
        // std::cout << "Custom Cube:\n" << customCube << std::endl;

        customCube.print("this isn the cube");
        return 0;

}