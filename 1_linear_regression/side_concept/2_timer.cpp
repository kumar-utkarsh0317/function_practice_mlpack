#include <iostream>
using namespace std;
#include "../MlPack/mlpack/build/installdir/include/mlpack/core.hpp"
#include "../MlPack/mlpack/build/installdir/include/mlpack/core/util/timers.hpp"


int main()
{
    mlpack::util::Timers timers;

    // Start the timer.
    timers.Start("MyTimer");

    // Code block to be timed.
    for (int i = 0; i < 1000000; ++i)
    {
        // Some computation...
    }

    // Stop the timer.
    timers.Start("MyTimer");

    // Rest of the code...
    // Print timing information.
    // timeval duration = timers.Get<decltype(std::chrono::microseconds)>("MyTimer");
    



    return 0;
}