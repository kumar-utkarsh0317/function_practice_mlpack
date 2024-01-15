
#ifndef ROUGH_HPP
#define ROUGH_HPP

#include <mlpack.hpp>
using namespace std;

template<typename T = int>
class Employee
{
        private:
        T age;
        T weight;
        arma::mat matrix;

        void print();

        private:
        Employee();
        Employee(T age, T weight);
        Employee(T age, T weight, arma::mat m);


          //! Load weights into the model.
        void LoadModel(const std::string& filePath);

        //! Save weights for the model.
        void SaveModel(const std::string& filePath);


};


#endif