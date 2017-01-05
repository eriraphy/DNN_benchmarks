#include "daal.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;


/*Usage:
convert colomn-domain matrix to row-domain matrix(mexArray to C Array)
input_matrix: colomn-domain; output_matrix: row-domain
matrix_cr_conv(double *cdmatrix, int ncol, int nrow)

convert row-domain matrix to colomn-domain matrix(C Array to mexArray)
input_matrix: row-domain; output_matrix: colomn-domain
matrix_cr_conv(double *rdmatrix, int nrow, int ncol)
*/

template<typename type>
void matrix_cr_conv(type *ipmatrix, size_t dx, size_t dy) {
	int i;
	int j;
	type *opmatrix;
	opmatrix = (type *)malloc(sizeof(type)*dx*dy);
	for (i = 0; i < dy; i++) {
		for (j = 0; j < dx; j++) {
			opmatrix[i*dx + j] = ipmatrix[i + j*dy];
		}
	}
	for (i = 0; i < dy*dx; i++) {
		ipmatrix[i] = opmatrix[i];
	}
	free(opmatrix);
	return;
}


/*Convert C matrix to DAAL Tensor*/
/*Input of 1D/2D/3D/4D matrix*/
template<typename type>
SharedPtr<data_management::Tensor> Matrix_Tensor(type *mxptr, size_t nrow, size_t ncol, size_t nm, size_t nc) {
	Collection<size_t> dims;
	dims.push_back(nrow);
	dims.push_back(ncol);
	dims.push_back(nm);
	dims.push_back(nc);
	data_management::HomogenTensor<type> *tensor = new data_management::HomogenTensor<type>(dims, mxptr);
	SharedPtr<data_management::Tensor> temp(tensor);
	return temp;
}

template<typename type>
SharedPtr<data_management::Tensor> Matrix_Tensor(type *mxptr, size_t nrow) {
	Collection<size_t> dims;
	dims.push_back(nrow);
	data_management::HomogenTensor<type> *tensor = new data_management::HomogenTensor<type>(dims, mxptr);
	SharedPtr<data_management::Tensor> temp(tensor);
	return temp;
}

/*Get the Pointer of a DAAL Tensor*/
template<typename type>
type *TensorPtr(SharedPtr<data_management::Tensor>input, size_t length) {
	data_management::SubtensorDescriptor<type> block;
	input->getSubtensor(0, 0, 0, length, readOnly, block);
	type *temp = block.getPtr();
	return temp;
}

/*Read Matrix from bin file*/
/*Binary storage; 
First item: numCols; 
Second item: numRows; 
Third item: ifistrans;
Remainder: Matrix data*/
template<typename type>
type *bin_read(const char* fileName)
{
	FILE* f = fopen(fileName, "rb");
	if (f == NULL)
	{
		printf("can not open %s\n", fileName);
	}
	bool istrans;
	long int numCols;
	long int numRows;
	fread(&numCols, sizeof(long int), 1, f);
	fread(&numRows, sizeof(long int), 1, f);
	fread(&istrans, sizeof(bool), 1, f);
	type *data;
	data = (type *)malloc(sizeof(type)*numCols*numRows);
	fread(&*data, sizeof(type), numCols*numRows, f);
	fclose(f);
	return data;
}