#include <iostream>
using namespace std;
#include<string>
// #include "../MlPack/mlpack/build/installdir/include/mlpack.hpp"

int *sumArray(int **array, int row, int col){

   
    for (int i = 0; i < col; i++)
    {
        for (int j = 1; j < row; j++)
        {
            array[0][i] += array[j][i];
        }
    }

    return array[0];
    
}

int main(){
    int array1[5] = {1, 2, 3, 4, 5};
    int array2[5] = {10, 20, 30, 40, 50};

    int *array_pointer[2] = {array1, array2};

    int *final_array_pointer = sumArray(array_pointer, 2, 5);

    for (int i = 0; i < 5; i++)
    {
        cout<<final_array_pointer[i]<<" "<<endl;
    }
    
    return 0;
}
