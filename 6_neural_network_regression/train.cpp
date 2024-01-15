// #define MLPACK_ENABLE_ANN_SERIALIZATION

#include "mlpack.hpp"
// #include <ensmallen.hpp>
using namespace std;


int main()
{
        /**
        * making a simple linear equation in two 
        * variable data set in which i can train my
        * linear regression model
        * y = mx + c
        * first column will be x
        * secod column will be y
        * 
        * m = 0.7 and c = 3
        */

       // creating a custom dataset and saving it in csv file
        string path = "/home/kumarutkarsh/Desktop/mlpack_function_practice/6_neural_network_regression/data.csv";
        arma::mat matrix(10000, 2);
        for (int i = 0; i < 10000; i++) 
        {
                matrix(i, 0) = i;
                matrix(i, 1) = 0.7 * i + 3;
        }
        matrix = matrix.t();
        mlpack::data::Save(path, matrix);

        // loading the dataset
        cout<< "Loading the data"<<endl;
        arma::mat data_set;
        bool load_status = mlpack::data::Load(path, data_set);

        if(!load_status)
        {
                return -1;
        }
        cout<< "data_set rows:: "<<data_set.n_rows<<endl;
        cout<< "data_set cols:: "<<data_set.n_cols<<endl;

        // spliting the data into training and validation dataset
        arma::mat train, valid;
        float RATIO = 0.3;
        mlpack::data::Split(data_set, train, valid, RATIO);

        arma::mat Xtrain, Xvalid, ytrain, yvalid;
        Xtrain = train.row(0);
        ytrain = train.row(1);

        // Xvalid and yvalid assignment corrected
        Xvalid = valid.row(0);
        yvalid = valid.row(1);

        /**
         * scaling can be done but i am not doing here
        */


        /**
         * creating the model
         * model will only one layer 
         * and optimizer have to change the parameter of only one weight and bias
        */
        mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
        // model.Add<mlpack::Linear>(3);
        // model.Add<mlpack::LeakyReLU>();
        // model.Add<mlpack::Linear>(2);
        // model.Add<mlpack::LeakyReLU>();
        model.Add<mlpack::Linear>(1);

        /**
         * telling the ffn that the input that we are giving it has only one feature in its one sample
        */
        vector<size_t> v= {1};
        model.InputDimensions() = v;   // it basically takes the input features in one data point
        model.Reset();


        /**
         * giving custom weights to out model
        */
        // cout<<"number of weights "<<model.WeightSize()<<endl;
        // arma::mat& parameters = model.Parameters();
        // parameters(0) = 0.7;
        // parameters(1) = 3.0;
        // model.Reset();


        cout<<"\n\n";
        model.Parameters().print("this is the parameter matrix");

        /**
         * seting parameters for the stochastic gradient descent SGD
        */
       // Number of epochs for training.
        int EPOCHS = 1;
        double STEP_SIZE = 0.01;
        constexpr int BATCH_SIZE = 100;
        constexpr double STOP_TOLERANCE = 1e-8;


        cout<< "inside the for loop"<<endl;

        for (int i = 0; i < 1000; i++)
        {
                
                // ens::Adam optimizer(
                //         0.05,    // Step size of the optimizer.
                //         32,      // Batch size. Number of data points that are used in each iteration.
                //         0.9,     // Exponential decay rate for the first moment estimates.
                //         0.999,   // Exponential decay rate for the weighted infinity norm estimates.
                //         1e-8,    // Value used to initialise the mean squared gradient parameter.
                //         Xtrain.n_cols * 1,  // Max number of iterations.
                //         1e-8,    // Tolerance.
                //         true);   // Shuffle the data before each epoch.

                ens::AdaDelta optimizer(0.05, 1, 0.99, 1e-8, Xtrain.n_cols, 1e-9, true);

                // ens::StandardSGD optimizer(0.01, 32, 100000, 1e-5, true);


                model.Train(Xtrain,
                        ytrain,
                        optimizer,
                        // PrintLoss Callback prints loss for each epoch.
                        ens::PrintLoss(),
                        // Progressbar Callback prints progress bar for each epoch.
                        ens::ProgressBar(),
                        // Stops the optimization process if the loss stops decreasing
                        // or no improvement has been made. This will terminate the
                        // optimization once we obtain a minima on training set.
                        ens::EarlyStopAtMinLoss(20)
                );
                // model.Reset();

                cout<<"\n\n";
                model.Parameters().print("this is the parameter matrix");


        }

        // arma::mat input = {1, 2, 3};
        // model.Predict(input).print("this is the predicted value");














        // arma::mat _ = {
        //         {1000, 1000},
        //         {1000, 1000},
        //         {1000, 1000}
        // };
        // arma::mat result;
        // model.Predict(_, result);
        // result.print("this is the result");

        return 0;

}