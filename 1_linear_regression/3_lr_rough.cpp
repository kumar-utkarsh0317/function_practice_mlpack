#include <iostream>
#include<string>
#include "/home/kumar/Desktop/mlpack_source_code_and_build/mlpack/build/installdir/include/mlpack.hpp"
#include <armadillo>
using namespace std;

//the agument in the main will help in taking input when we are running the executable file
//arg c=>number of input given at the runtime of executable 
//argv is a pointer list that contains the location of the input string
int main(int argc, char**argv){
    //here we want two argument from the command line
    //first ./executable and second file.csv
    //>>>> ./executable file.csv
    if(argc != 2){
        std::cerr<<"usage: "<<argv[0]<<"<csv_training>"<<endl;
        return -1;
    }

    //loading the data 
    std::cout<<"we are loading the data from the file::"<<argv[1]<<std::endl;
    arma::mat predictor;
    //declearing the field without any argument so any number of element it can store
    arma::field<std::string> headers;
    //data from the first col of the csv file will be stored in the header and rest of the data
    //will be used in the predictors
    //mlpack depends on armadillo and armadillo depends on column major policy so we 
    //are transposing the csv data
    bool load_status = predictor.load(arma::csv_name(argv[1], headers, arma::csv_opts::trans));
    cout<<"predictor rows and cols are ("<<predictor.n_rows<<", "<<predictor.n_cols<<")\n\n";
    cout<<"this is the header that we are getting\n"<<headers<<"\n\n";
    //if unable to load the file then terminate the program
    if(!load_status){
        cout<<"file "<<argv[1]<<"could not be loaded\n";
        return 1;
    }
    std::cout<<"file loaded. and the headers are::"<<headers<<std::endl;
    // column major matrix -> last row contains the responses
    //output variable
    arma::rowvec responses = predictor.row(predictor.n_rows - 1);
    // input variable
    predictor.shed_row(predictor.n_rows - 1);

    //training the model
    std::cout<<"we are now training our model"<<std::endl;
    mlpack::LinearRegression lr(predictor, responses);
    std::cout << "Computed error: "<< lr.ComputeError(predictor, responses) << '\n';

    //saving the model to model.xml
    std::cout<<"now we are saving the model...\n";
    if (!mlpack::data::Save("3_lr_rough.xml", "model_rough", lr)){
        std::cerr<< "could not save the model to model.xml file \n";
        return 1;
    }
    std::cout<<"the model data is saved to model.xml \n";
    cout<<"********************************train_complete************************\n\n";
    return 0;
}
