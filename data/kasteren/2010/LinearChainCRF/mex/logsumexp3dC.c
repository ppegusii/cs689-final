#include "mex.h"
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
 const int *dims;
 int i, j, m, n, d,ndim;
 double *inData,*outData,*maxVal;
 mxArray *maxDoubleMat;
 if (nlhs!= 1 | nrhs > 2)
{
	mexErrMsgTxt("The number of input should be 1 and output arguments should be 1");
}
ndim = mxGetNumberOfDimensions(prhs[0]);
dims = mxGetDimensions(prhs[0]);

if (ndim!=3)
{
	mexErrMsgTxt("The number of dimensions should be 3 for this function.");
}
 
plhs[0] = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
inData = mxGetPr(prhs[0]);
outData = mxGetPr(plhs[0]);
	 
maxDoubleMat = mxCreateDoubleMatrix(dims[0]*dims[1],1, mxREAL);
maxVal = mxGetPr(maxDoubleMat);

for (i = 0; i < dims[0]*dims[1]; i++)
{
	for (j = 0; j < dims[2]; j++)
	{
		if (j==0)
		{
			maxVal[i] = inData[j*dims[0]*dims[1]];
		}
		else
		{
			if (max(maxVal[i],inData[(j*dims[0]*dims[1])+i]))
				maxVal[i] = inData[(j*dims[0]*dims[1])+i];
		}
	}
}

for (i = 0; i < dims[0]*dims[1]; i++)
{
	for (j = 0; j < dims[2]; j++)
	{
		if (mxIsInf(maxVal[i])!=true)
		{
			outData[i] += exp(inData[j*dims[0]*dims[1]+i] - maxVal[i]);
		}
	}
	outData[i] = log(outData[i]) + maxVal[i];
}
}
             