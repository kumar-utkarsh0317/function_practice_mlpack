#ifndef MODELS_MODELS_DARKNET_DARKNET_HPP
#define MODELS_MODELS_DARKNET_DARKNET_HPP

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

namespace mlpack
{
        namespace models
        {

                // class defination starts from here
                template <
                    typename OutputLayerType = CrossEntropyError<>,
                    typename InitializationRuleType = RandomInitialization,
                    size_t DarkNetVersion = 19>
                class DarkNet
                {
                public:
                        DarkNet(const size_t inputChannel,
                                const size_t inputWidth,
                                const size_t inputHeight,
                                const size_t numClasses = 1000,
                                const std::string &weights = "none",
                                const bool includeTop = true);

                        /**
                         * DarkNet constructor intializes input shape and number of classes.
                         *
                         * @param inputShape A three-valued tuple indicating input shape.
                         *     First value is number of channels (channels-first).
                         *     Second value is input height. Third value is input width.
                         * @param numClasses Optional number of classes to classify images into,
                         *     only to be specified if includeTop is  true.
                         * @param weights One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
                         */
                        DarkNet(const std::tuple<size_t, size_t, size_t> inputShape,
                                const size_t numClasses = 1000,
                                const std::string &weights = "none",
                                const bool includeTop = true);

                        //! Get Layers of the model.
                        FFN<OutputLayerType, InitializationRuleType> &GetModel()
                        {
                                return darkNet;
                        }

                        //! Load weights into the model.
                        void LoadModel(const std::string &filePath);

                        //! Save weights for the model.
                        void SaveModel(const std::string &filePath);

                private:
                        /**
                         * Adds Convolution Block.
                         *
                         * @tparam SequentialType Layer type in which convolution block will
                         *     be added.
                         *
                         * @param inSize Number of input maps.
                         * @param outSize Number of output maps.
                         * @param kernelWidth Width of the filter/kernel.
                         * @param kernelHeight Height of the filter/kernel.
                         * @param strideWidth Stride of filter application in the x direction.
                         * @param strideHeight Stride of filter application in the y direction.
                         * @param padW Padding width of the input.
                         * @param padH Padding height of the input.
                         * @param batchNorm Boolean to determine whether a batch normalization
                         *     layer is added.
                         * @param negativeSlope Negative slope hyper-parameter for LeakyReLU.
                         * @param baseLayer Layer in which Convolution block will be added, if
                         *                  NULL added to darkNet FFN.
                         */
                        template <typename SequentialType = Sequential<>>
                        void ConvolutionBlock(const size_t inSize,   //in channels
                                              const size_t outSize,  //out channels
                                              const size_t kernelWidth,
                                              const size_t kernelHeight,
                                              const size_t strideWidth = 1,
                                              const size_t strideHeight = 1,
                                              const size_t padW = 0,
                                              const size_t padH = 0,
                                              const bool batchNorm = true,
                                              const double negativeSlope = 1e-1,
                                              SequentialType *baseLayer = NULL)
                        {

                                //bottleNeck has the address of a sequential class present in heap
                                Sequential<> *bottleNeck = new Sequential<>();
                                //The new Convolution<>(...) creates a dynamically allocated instance of the Convolution<> layer on the heap, and the pointer to this instance is passed to the Add method.
                                // Add function may be defined in such a way that it can also work with the pointer of the object as well the object it self 
                                bottleNeck->Add(new Convolution<>(inSize, outSize, kernelWidth,
                                                                  kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
                                                                  inputHeight));

                                // Update inputWidth and input Height.
                                mlpack::Log::Info << "Conv Layer.  ";
                                mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight << ", " << inSize << ") ----> ";

                                //after each convalution operation inputweidth and inputheight will be updated
                                inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
                                inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
                                mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight << ", " << outSize << ")" << std::endl;

                                if (batchNorm)
                                        bottleNeck->Add(new BatchNorm<>(outSize, 1e-5, false));

                                bottleNeck->Add(new LeakyReLU<>(negativeSlope));

                                if (baseLayer != NULL)
                                        baseLayer->Add(bottleNeck);
                                else
                                        darkNet.Add(bottleNeck);
                        }

                        /**
                         * Adds Pooling Block.
                         *
                         * @param factor The factor by which input dimensions will be divided.
                         * @param type One of "max" or "mean". Determines whether add mean pooling
                         *     layer or max pooling layer.
                         */
                        void PoolingBlock(const size_t factor = 2,
                                          const std::string type = "max")
                        {
                                if (type == "max")
                                {
                                        darkNet.Add(new AdaptiveMaxPooling<>(
                                            std::ceil(inputWidth * 1.0 / factor),
                                            std::ceil(inputHeight * 1.0 / factor)));
                                }
                                else
                                {
                                        darkNet.Add(new AdaptiveMeanPooling<>(std::ceil(inputWidth * 1.0 /
                                                                                        factor),
                                                                              std::ceil(inputHeight * 1.0 / factor)));
                                }

                                mlpack::Log::Info << "Pooling Layer.  ";
                                mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight << ") ----> ";

                                // Update inputWidth and inputHeight.
                                inputWidth = std::ceil(inputWidth * 1.0 / factor);  //again updating the input width and input height
                                inputHeight = std::ceil(inputHeight * 1.0 / factor);
                                mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight << ")" << std::endl;
                        }

                        /**
                         * Adds bottleneck block for DarkNet 19.
                         *
                         * It's represented as:
                         * ConvolutionLayer(inputChannel, inputChannel * 2, stride)
                         *           |
                         * ConvolutionLayer(inputChannel * 2, inputChannel, 1)
                         *           |
                         * ConvolutionLayer(inputChannel, inputChannel * 2, stride)
                         *
                         * @param inputChannel Input channel in the convolution block.
                         * @param kernelWidth Width of the filter/kernel.
                         * @param kernelHeight Height of the filter/kernel.
                         * @param padWidth Padding in convolutional layer.
                         * @param padHeight Padding in convolutional layer.
                         */
                        void DarkNet19SequentialBlock(const size_t inputChannel,
                                                      const size_t kernelWidth,
                                                      const size_t kernelHeight,
                                                      const size_t padWidth,
                                                      const size_t padHeight)
                        {
                                ConvolutionBlock(inputChannel, inputChannel * 2,
                                                 kernelWidth, kernelHeight, 1, 1, padWidth, padHeight, true);
                                ConvolutionBlock(inputChannel * 2, inputChannel,
                                                 1, 1, 1, 1, 0, 0, true);
                                ConvolutionBlock(inputChannel, inputChannel * 2,
                                                 kernelWidth, kernelHeight, 1, 1, padWidth, padHeight, true);
                        }

                        /**
                         * Adds residual bottleneck block for DarkNet 53.
                         *
                         * @param inputChannel Input channel in the bottle-neck.
                         * @param kernelWidth Width of the filter/kernel.
                         * @param kernelHeight Height of the filter/kernel.
                         * @param padWidth Padding in convolutional layer.
                         * @param padHeight Padding in convolutional layer.
                         */
                        void DarkNet53ResidualBlock(const size_t inputChannel,
                                                    const size_t kernelWidth = 3,
                                                    const size_t kernelHeight = 3,
                                                    const size_t padWidth = 1,
                                                    const size_t padHeight = 1)
                        {
                                mlpack::Log::Info << "Residual Block Begin." << std::endl;
                                Residual<> *residualBlock = new Residual<>();
                                ConvolutionBlock(inputChannel, inputChannel / 2,
                                                 1, 1, 1, 1, 0, 0, true, 1e-2, residualBlock);
                                ConvolutionBlock(inputChannel / 2, inputChannel, kernelWidth,
                                                 kernelHeight, 1, 1, padWidth, padHeight, true, 1e-2, residualBlock);
                                darkNet.Add(residualBlock);
                                mlpack::Log::Info << "Residual Block end." << std::endl;
                        }

                        /**
                         * Return the convolution output size.
                         *
                         * @param size The size of the input (row or column).
                         * @param k The size of the filter (width or height).
                         * @param s The stride size (x or y direction).
                         * @param padding The size of the padding (width or height) on one side.
                         * @return The convolution output size.
                         */
                        size_t ConvOutSize(const size_t size,  //size of height or can be width,,,, can be used for both
                                           const size_t k,
                                           const size_t s,
                                           const size_t padding)
                        {
                                return std::floor(size + 2 * padding - k) / s + 1;
                        }

                        //! Locally stored DarkNet Model.
                        FFN<OutputLayerType, InitializationRuleType> darkNet;

                        //! Locally stored width of the image.
                        size_t inputWidth;

                        //! Locally stored height of the image.
                        size_t inputHeight;

                        //! Locally stored number of channels in the image.
                        size_t inputChannel;

                        //! Locally stored number of output classes.
                        size_t numClasses;

                        //! Locally stored type of pre-trained weights.
                        std::string weights;
                }; // DarkNet class.

                // Convenience typedefs for different DarkNet models.
                typedef DarkNet<CrossEntropyError<>, RandomInitialization, 19>
                    DarkNet19;

                typedef DarkNet<CrossEntropyError<>, RandomInitialization, 53>
                    DarkNet53;

        } // namespace models
} // namespace mlpack

#include "darknet_impl.hpp"

#endif