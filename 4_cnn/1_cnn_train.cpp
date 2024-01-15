#include<mlpack.hpp>
using namespace std;

//i think mlpack provide the values of these macros
#if ((ENS_VERSION_MAJOR < 2) || ((ENS_VERSION_MAJOR == 2) && (ENS_VERSION_MINOR < 13)))
  #error "need ensmallen version 2.13.0 or later"
#endif




int main()
{
    cout<<"\n\n";
    // constexpr is a keyword used to indicate that a variable, function, or expression can be evaluated at compile time
    const double RATIO = 0.1;

    // Allow 60 passes over the training data, unless we are stopped early by
    // EarlyStopAtMinLoss.
    const int EPOCHS = 60;

    // Number of data points in each iteration of SGD.
    const int BATCH_SIZE = 50;

    // Step size of the optimizer.
    const double STEP_SIZE = 1.2e-3;

    // Labeled dataset that contains data for training is loaded from CSV file.
    // Rows represent features, columns represent data points.
    arma::mat dataset;
    string csv_path = "/home/kumarutkarsh/Desktop/mlpack_function_practice/datasets/mnist_dataset/train.csv";
    mlpack::data::Load(csv_path, dataset);

    //in dataset
    //rows ==> features
    //cols ==> sample points
    cout<<"for dataset rows:: "<<dataset.n_rows<<" cols::"<<dataset.n_cols<<"\n";

    //creating the train and test split
    arma::mat train, test;
    mlpack::data::Split(dataset, train, test, RATIO);
    mlpack::data::Save("train.csv", train);  //before saving it will be first transformed
    mlpack::data::Save("test.csv", test);

    cout<<"for train rows:: "<<train.n_rows<<" cols::"<<train.n_cols<<"\n";
    cout<<"for test rows:: "<<test.n_rows<<" cols::"<<test.n_cols<<"\n";

  // The train and valid datasets contain both - the features as well as the
  // class labels. Split these into separate mats.
  const arma::mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) /
      256.0;
  const arma::mat testX = test.submat(1, 0, test.n_rows - 1, test.n_cols - 1) /
      256.0;

  // Labels should specify the class of a data point and be in the interval [0,
  // numClasses).

  // Create labels for training and testatiion datasets.
  const arma::mat trainY = train.row(0);
  const arma::mat testY = test.row(0);

  // Specify the NN model. NegativeLogLikelihood is the output layer that
  // is used for classification problem. RandomInitialization means that
  // initial weights are generated randomly in the interval from -1 to 1.
  mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization> model;

  
  // Add the first convolution layer.
  model.Add<mlpack::Convolution>(6,  // Number of filters
                         5,  // Filter width._
                         5,  // Filter height.
                         1,  // Stride along width.
                         1,  // Stride along height.
                         0,  // Padding width.
                         0   // Padding height.
  );
  // Add first ReLU.
  model.Add<mlpack::LeakyReLU>();

  // Add first pooling layer. Pools over 2x2 fields in the input.
  model.Add<mlpack::MaxPooling>(2, // Width of field.
                        2, // Height of field.
                        2, // Stride along width.
                        2, // Stride along height.
                        true);

  // Add the second convolution layer.
  model.Add<mlpack::Convolution>(16, // Number of output activation maps.
                         5,  // Filter width.
                         5,  // Filter height.
                         1,  // Stride along width.
                         1,  // Stride along height.
                         0,  // Padding width.
                         0   // Padding height.
  );

  // Add the second ReLU.
  model.Add<mlpack::LeakyReLU>();

  // Add the second pooling layer.
  model.Add<mlpack::MaxPooling>(2, 2, 2, 2, true);
  
  // Add the final dense layer.
  model.Add<mlpack::Linear>(10);   //may be mlpack will convert the 3d matrix in the linear form 
  model.Add<mlpack::LogSoftMax>();


  // inputdimensions returns the refrence to the vector that stores the dimensio of the input layer
  model.InputDimensions() = vector<size_t>({ 28, 28 });





  cout<<"///////////printing the shape of the model"<<"--------"<<endl;
  cout<<"number of rows::"<<model.n_rows<<"number of cols::"<<model.n_cols<<endl;
  cout << "Start training ..." << endl;

  //set parameters for the adam optimizer
  ens::Adam optimizer(
    STEP_SIZE,  // Step size of the optimizer.
    BATCH_SIZE, // Batch size. Number of data points that are used in each
                // iteration.
    0.9,        // Exponential decay rate for the first moment estimates.
    0.999, // Exponential decay rate for the weighted infinity norm estimates.
    1e-8,  // Value used to initialise the mean squared gradient parameter.
    EPOCHS * trainX.n_cols, // Max number of iterations.
    1e-8,           // Tolerance.
    true);

  // ens::adaDelta optimizer(
  //   STEP_SIZE,
  //   BATCH_SIZE,
  //   0.95,
  //   1e-8,  // Value used to initialise the mean squared gradient parameter.
  //   EPOCHS * trainX.n_cols, // Max number of iterations.
  //   1e-8,           // Tolerance.
  //   true);



  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              // Stop the training using Early Stop at min loss.
              ens::EarlyStopAtMinLoss(
                  [&](const arma::mat& /* param */)
                  {
                    double validationLoss = model.Evaluate(testX, testY);
                    cout << "Validation loss: " << validationLoss << "."
                        << endl;
                    return validationLoss;
                  }));


  // Matrix to store the predictions on train and validation datasets.
  arma::mat predOut;
  // Get predictions on training data points.
  model.Predict(trainX, predOut);

  predOut.print("this is the prediction made by our model");



    return 0;
}