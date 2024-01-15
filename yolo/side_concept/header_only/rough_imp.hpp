#ifndef ROUGH_IMP_HPP
#define ROUGH_IMP_HPP

#include "rough.hpp"



/*The template<typename T> part is needed in the implementation file to inform the compiler that 
the following code is a template specialization for the Employee class with the template parameter T.

Default template arguments are typically specified in the header file where the template is declared.
*/
template<typename T>
Employee<T>::Employee() : age(0), weight(0), matrix({{1, 2, 3}})
{
        //nothing here
        cout<<"inside the default constructor"<<endl;
}


template<typename T>
Employee<T>::Employee(T age, T weight, arma::mat matrix) : Employee(age, weight)
{
        cout<<"inside the constructor 1"<<endl;
        arma::mat m = {{1, 2, 3},
                        {4, 5, 6},
                        {7, 8, 9}};
        this->matrix = m;
}

template<typename T>
Employee<T>::Employee(T age, T weight) : age(age), weight(weight)
{
        cout<<"inside the constructor 2"<<endl;
        arma::mat m = {{1, 2, 3},
                        {4, 5, 6}};
        this->matrix = arma::mat(m);
}


template<typename T>
void Employee<T>::print()
{
        cout<<"age: "<<age<<"weight: "<<weight<<endl;
        matrix.print("this is the matrix");
}




///implementing the load and save model
template<typename T>
void Employee<T>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "yolo" + yoloVersion, yolo);
  Log::Info << "Loaded model." << std::endl;
}

template<typename T>
void Employee<T>::SaveModel(const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "yolo" + yoloVersion, yolo);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

#endif