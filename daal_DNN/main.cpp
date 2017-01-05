#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "time.h"
#include "neural_network_service.h"
//#include "mkl_dnn.h"
#include "daal.h"

#define dimension (4)

void test_daal()
{
	services::Environment::getInstance()->setNumberOfThreads(2);
	int i;
	/*Read Data fromr bin file*/
	float *data_input = bin_read<float>("data/input_data.bin");
	float *data_conv = bin_read<float>("data/conv.bin");
	float *data_bias = bin_read<float>("data/base.bin");

	/*Convert data into DAAL tensor*/
	SharedPtr<Tensor> inputdata = Matrix_Tensor<float>(data_input, 1, 3, 181, 181);
	SharedPtr<Tensor> inputweight = Matrix_Tensor<float>(data_conv, 64, 3, 7, 7);
	SharedPtr<Tensor> inputbias = Matrix_Tensor<float>(data_bias, 64);

	/*Set up Layer*/
	convolution2d::forward::Batch<> convolution2dLayerForward;

	/*Define Input and Parameters*/
	convolution2dLayerForward.input.set(forward::data, inputdata);
	convolution2dLayerForward.input.set(forward::weights, inputweight);
	convolution2dLayerForward.input.set(forward::biases, inputbias);

	convolution2dLayerForward.parameter.nGroups = 1;
	convolution2dLayerForward.parameter.kernelSizes = convolution2d::KernelSizes(7, 7);

	convolution2dLayerForward.parameter.nKernels = 64;
	convolution2dLayerForward.parameter.paddings = convolution2d::Paddings(3, 3);
	convolution2dLayerForward.parameter.strides = convolution2d::Strides(2, 2);

	/*Start Compute*/
	clock_t start, end;
	start = clock();
	for (i = 0; i < 100; i++) {
	convolution2dLayerForward.compute();
	}
	end = clock();

	services::SharedPtr<convolution2d::forward::Result> convResult = convolution2dLayerForward.getResult();



	//float *resultValue = TensorPtr<float>(convResult->get(forward::value), 1);
	//float *test = TensorPtr<float>(inputweight, 3*7*7);
	services::Collection<size_t> wdim = convResult->get(convolution2d::auxWeights)->getDimensions();
	services::Collection<size_t> ddim = convResult->get(convolution2d::auxData)->getDimensions();
	services::Collection<size_t> dim = convResult->get(forward::value)->getDimensions();

	convResult->get(forward::resultForBackward);

	//printf("DAAL results:\n");

	//printf("First 20 results:\n");
	//for (i = 0; i < 20; i++) {
	//	std::cout << resultValue[i] << '\n';

	//}

	//printf("Second 20 results:\n");
	//for (i = 0; i < 20; i++) {
	//	std::cout << resultValue[i + 91 * 91] << '\n';

	//}

	//std::cout << "weight dims" << "\n";
	//std::cout << wdim[0] << "\n" << wdim[1] << "\n" << wdim[2] << "\n" << wdim[3] << "\n";
	//std::cout << "data dims" << "\n";
	//std::cout << ddim[0] << "\n" << ddim[1] << "\n" << ddim[2] << "\n" << ddim[3] << "\n";
	//std::cout << "result dims" << "\n";
	//std::cout << dim[0] << "\n" << dim[1] << "\n" << dim[2] << "\n" << dim[3] << "\n";
	printf("==DAAL_batch_Elapsed_Time %f ms==\n", (float)(end - start) / CLOCKS_PER_SEC);

	//FILE* f = fopen("daal_test.txt", "w");
	//if (f != NULL)
	//{
	//	fprintf(f, "First 5000 results:\n");
	//	for (long int i = 0; i < 5000; i++)
	//	{
	//		fprintf(f, "%f\n", resultValue[i]);
	//	}
	//	fclose(f);
	//}
	//system("pause");

	return;
}

//static dnnError_t init_conversion(dnnPrimitive_t *cv, float **ptr_out,
//	dnnLayout_t lt_pr, dnnLayout_t lt_us, float *ptr_us)
//{
//	dnnError_t err;
//	*ptr_out = NULL;
//	if (!dnnLayoutCompare_F32(lt_pr, lt_us)) {
//		dnnConversionCreate_F32(cv, lt_us, lt_pr);
//		dnnAllocateBuffer_F32((void**)ptr_out, lt_pr);
//	}
//	else {
//		*ptr_out = ptr_us;
//	}
//	return E_SUCCESS;
//
//bail_out:
//	if (*ptr_out) dnnReleaseBuffer_F32(*ptr_out);
//	return err;
//}
//
//void test_mkl()
//{
//	int i;
//	dnnError_t err;
//	size_t batch_size = 1;
//
//	size_t outputSize[dimension] = { 91, 91, 64, batch_size };
//	size_t outputStrides[dimension] = { 1, 91, 91 * 91, 91 * 91 * 64 };
//
//	size_t inputSize[dimension] = { 181, 181, 3, batch_size };
//	size_t inputStrides[dimension] = { 1, 181, 181 * 181, 181 * 181 * 3 };
//
//	size_t filterSize[dimension] = { 7, 7, 3, 64 };
//	size_t filterStrides[dimension] = { 1, 7, 7 * 7, 7 * 7 * 3 };
//
//	size_t convolutionStride[dimension - 2] = { 2, 2 };
//	int inputOffset[dimension - 2] = { -3, -3 };
//
//	size_t biasSize[1] = { outputSize[2] };
//	size_t biasStrides[1] = { 1 };
//
//	dnnLayout_t lt_user_input = NULL,
//		lt_user_filt = NULL,
//		lt_user_bias = NULL,
//		lt_user_output = NULL;
//	dnnPrimitive_t conv1 = NULL;
//	dnnLayout_t lt_conv1_input = NULL,
//		lt_conv1_filt = NULL,
//		lt_conv1_bias = NULL,
//		lt_conv1_output = NULL;
//	float* resConv1[dnnResourceNumber] = { 0 };
//	dnnPrimitive_t cv_user_to_conv1_input = NULL,
//		cv_user_to_conv1_filt = NULL,
//		cv_user_to_conv1_bias = NULL,
//		cv_conv1_to_user_output = NULL;
//	dnnPrimitiveAttributes_t attributes = NULL;
//
//	float *data_input = bin_read<float>("data/input_data.bin");
//	float *data_conv = bin_read<float>("data/conv.bin");
//	float *data_bias = bin_read<float>("data/base.bin");
//	float *data_output = NULL;
//
//	dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides);
//	dnnLayoutCreate_F32(&lt_user_filt, dimension, filterSize, filterStrides);
//	dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides);
//	dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides);
//
//	dnnPrimitiveAttributesCreate_F32(&attributes);
//
//	dnnConvolutionCreateForwardBias_F32(&conv1, attributes,
//		dnnAlgorithmConvolutionDirect, dimension, inputSize,
//		outputSize, filterSize, convolutionStride, inputOffset,
//		dnnBorderZeros);
//
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv1, dnnResourceSrc);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv1, dnnResourceFilter);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, conv1, dnnResourceBias);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output, conv1, dnnResourceDst);
//
//	init_conversion(&cv_user_to_conv1_input, &resConv1[dnnResourceSrc], lt_conv1_input, lt_user_input, data_input);
//	init_conversion(&cv_user_to_conv1_filt, &resConv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt, data_conv);
//	init_conversion(&cv_user_to_conv1_bias, &resConv1[dnnResourceBias], lt_conv1_bias, lt_user_bias, data_bias);
//	dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDst], lt_conv1_output);
//
//	init_conversion(&cv_conv1_to_user_output, &data_output, lt_user_output, lt_conv1_output, resConv1[dnnResourceDst]);
//
//	if (cv_user_to_conv1_filt) dnnConversionExecute_F32(cv_user_to_conv1_filt, data_conv, resConv1[dnnResourceFilter]);
//	if (cv_user_to_conv1_bias) dnnConversionExecute_F32(cv_user_to_conv1_bias, data_bias, resConv1[dnnResourceBias]);
//	if (cv_user_to_conv1_input) dnnConversionExecute_F32(cv_user_to_conv1_input, data_input, resConv1[dnnResourceSrc]);
//
//	clock_t start, end;
//	start = clock();    
//
//	//for (int i = 0; i < 100; i++) {
//		dnnExecute_F32(conv1, (void**)resConv1);
//	//}
//	end = clock();   
//	if (cv_conv1_to_user_output) dnnConversionExecute_F32(cv_conv1_to_user_output, resConv1[dnnResourceDst], data_output);
//
//	printf("MKL results:\n");
//	printf("First 20 results:\n");
//	for (i = 0; i < 20; i++) {
//		std::cout << data_output[i] << '\n';
//
//	}
//	printf("Second 20 results:\n");
//	for (i = 0; i < 20; i++) {
//		std::cout << data_output[i + 91 * 91] << '\n';
//
//	}
//
//	printf("==MKL_DNN_Elapsed_Time %f ms==\n", (float)(end - start) / CLOCKS_PER_SEC);
//
//	return;
//}


void main()
{
	test_daal();
	//printf(">>>>>>>>>>>>>>>>>>>==============<<<<<<<<<<<<<<<<<<<<\n");
	//test_mkl();


	return;
}
