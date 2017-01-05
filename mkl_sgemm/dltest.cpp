#include <math.h>
#include <time.h>
#include "dltest.h"
#include "mkl_dnn.h"

#define CUBLAS_gemm cblas_sgemm
#define CUblasNoTrans CblasNoTrans
#define CUblasTrans CblasTrans

#define dimension (4)

int im2col_cpu(const float* data_im, const int channels, const int height, const int width, const int ksize, const int pad,	const int stride, float* data_col)
{
	int height_col = (height - 2 * pad - ksize) / stride + 1;
	int width_col = (width - 2 * pad - ksize) / stride + 1;
	int channels_col = channels * ksize * ksize;
		for (int c = 0; c < channels_col; ++c){
			int w_offset = c % ksize;
			int h_offset = (c / ksize) % ksize;
			int c_im = c / ksize / ksize;
			for (int h = 0; h < height_col; ++h){
				for (int w = 0; w < width_col; ++w){
					int h_pad = h * stride + pad + h_offset;
					int w_pad = w * stride + pad + w_offset;
					if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width){
						*data_col=data_im[(c_im * height + h_pad) * width + w_pad];
					}else{
						*data_col=0;
					}
					data_col++;
				}
			}
		}
	return height_col*width_col;
}

void conv(Matrix* dataIn,Matrix* dataBuff,Matrix* dataOut,Matrix* weight,int inSz,int outSz,int pad,int stride,int numImg)
{
	int numColor=dataIn->getNumRows()/inSz/inSz;
	int numFilter=weight->getNumCols();
	float ss=sqrt(double(weight->getNumRows()/numColor)+0.1);
	int filterSz=sqrt(double(weight->getNumRows()/numColor)+0.1);
	assert(filterSz*filterSz*numColor==weight->getNumRows());
	dataBuff->resize(outSz*outSz,weight->getNumRows());
	dataBuff->setTrans(true);
	dataOut->resize(outSz*outSz*weight->getNumCols(),numImg);
	dataOut->setTrans(true);
	for (int i=0;i<numImg;i++){
		im2col_cpu(dataIn->getData()+i*dataIn->getNumRows(),numColor,inSz,inSz,filterSz,pad,stride,dataBuff->getData());
		cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,outSz*outSz,weight->getNumCols(),weight->getNumRows(),
			1,dataBuff->getData(),outSz*outSz,weight->getData(),weight->getNumRows(),0,dataOut->getData()+i*outSz*outSz*weight->getNumCols(),outSz*outSz);

	}
}

void addVec(Matrix* dataIn,Matrix* weightNorm){
	int C=weightNorm->getNumCols();
	for (int i=0;i<dataIn->getNumCols();i++)	{
		Matrix A(dataIn->getData()+i*dataIn->getNumRows(),dataIn->getNumRows()/C,C,true);
		A.addVector(weightNorm);
	}
}

void test_ours()
{
	int input_size = 181;
	int output_size = 91;
	// load data and filters
	Matrix* data_in = new Matrix("data/input_data.bin");
	Matrix* convWeight = new Matrix("data/conv.bin");
	Matrix* convBase = new Matrix("data/base.bin");

	Matrix* tempdata = new Matrix;
	Matrix* data_out = new Matrix;

	clock_t start, end;
	start = clock();    //开始时间

	// test
	for (int i=0; i<100;i++)
	{
	conv(data_in, tempdata, data_out, convWeight, input_size, output_size, -3, 2, 1); // kernel size 7x7
	addVec(data_out, convBase);
	}

	end = clock();    //结束时间
	printf("==MKL_sgemm_Elapsed_Time %f ms==\n", (float)(end - start) / CLOCKS_PER_SEC);


	// save data_out to txt
	FILE* f = fopen("mkl_sgemm.txt", "w");
	if (f != NULL)
	{
		fprintf(f, "First 5000 results:\n");
		Matrix a=&(*data_out);
		
		for (long int i = 0; i < a.getNumCols(); i++)
		{
			for (long int j = 0; j < a.getNumRows(); j++)
			{
				float dat = a(j, i);
			    fprintf(f, "%f\n", dat);

			}
		}
		fclose(f);
	}

	delete data_in;
	delete data_out;
	delete tempdata;
	delete convWeight;
	delete convBase;
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

//void test_mkldnn()
//{
//	// load data and filters
//	Matrix* data_in = new Matrix("data/input_data.bin");
//	Matrix* convWeight = new Matrix("data/conv.bin");
//	Matrix* convBase = new Matrix("data/base.bin");
//
//
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
//	float *user_i = NULL,
//		*user_c1_f = NULL,
//		*user_c1_b = NULL,
//		*user_o = NULL;
//
//
//	/*** data allocation ***/
//	user_i = (float*)malloc(sizeof(float)*(batch_size * 3 * 181 * 181));
//	user_c1_f = (float*)malloc(sizeof(float)*(64 * 3 * 7 * 7));
//	user_c1_b = (float*)malloc(sizeof(float)*(64));
//	// load data and filters
//	data_in->copyMem(user_i);
//	convWeight->copyMem(user_c1_f);
//	convBase->copyMem(user_c1_b);
//
//
//	/*** User's data description ***/
//	dnnLayoutCreate_F32(&lt_user_input, dimension, inputSize, inputStrides);
//	dnnLayoutCreate_F32(&lt_user_filt, dimension, filterSize, filterStrides);
//	dnnLayoutCreate_F32(&lt_user_bias, 1, biasSize, biasStrides);
//	dnnLayoutCreate_F32(&lt_user_output, dimension, outputSize, outputStrides);
//
//	/* Initialize attributes */
//	dnnPrimitiveAttributesCreate_F32(&attributes);
//
//	/*** convolution section ***/
//    dnnConvolutionCreateForwardBias_F32(&conv1, attributes,
//		dnnAlgorithmConvolutionDirect, dimension, inputSize,
//			outputSize, filterSize, convolutionStride, inputOffset,
//			dnnBorderZeros);
//
//
//	// Convolution describes what layout it expects
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_input, conv1, dnnResourceSrc);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_filt, conv1, dnnResourceFilter);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_bias, conv1, dnnResourceBias);
//	dnnLayoutCreateFromPrimitive_F32(&lt_conv1_output, conv1, dnnResourceDst);
//
//	init_conversion(&cv_user_to_conv1_input, &resConv1[dnnResourceSrc], lt_conv1_input, lt_user_input, user_i);
//	init_conversion(&cv_user_to_conv1_filt, &resConv1[dnnResourceFilter], lt_conv1_filt, lt_user_filt, user_c1_f);
//	init_conversion(&cv_user_to_conv1_bias, &resConv1[dnnResourceBias], lt_conv1_bias, lt_user_bias, user_c1_b);
//	dnnAllocateBuffer_F32((void**)&resConv1[dnnResourceDst], lt_conv1_output);
//
//	init_conversion(&cv_conv1_to_user_output, &user_o, lt_user_output, lt_conv1_output, resConv1[dnnResourceDst]);
//
//	/*** Execution ***/
//	if (cv_user_to_conv1_filt) dnnConversionExecute_F32(cv_user_to_conv1_filt, user_c1_f, resConv1[dnnResourceFilter]);
//	if (cv_user_to_conv1_bias) dnnConversionExecute_F32(cv_user_to_conv1_bias, user_c1_b, resConv1[dnnResourceBias]);
//	if (cv_user_to_conv1_input) dnnConversionExecute_F32(cv_user_to_conv1_input, user_i, resConv1[dnnResourceSrc]);
//
//
//	clock_t start, end;
//	start = clock();    //开始时间
//
//	// test
//	for(int i=0;i<100;i++)
//    dnnExecute_F32(conv1, (void**)resConv1);
//
//	end = clock();    //结束时间
//	printf("DNN batched 执行时间(秒)：%f\n", (double)(end - start) / CLOCKS_PER_SEC);
//
//
//    if (cv_conv1_to_user_output) dnnConversionExecute_F32(cv_conv1_to_user_output, resConv1[dnnResourceDst], user_o);
//	// save user_o to txt
//	FILE* f = fopen("mkl_data_out.txt", "w");
//	if (f != NULL)
//	{
//		for (int i = 0; i < 91 * 91 * 64; i++)
//		{
//			fprintf(f,"%f\n",user_o[i]);
//		}
//		fclose(f);
//	}
//}
