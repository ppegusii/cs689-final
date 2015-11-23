#include "mex.h"
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void logSumWithMax(double *inData, double maxVal, int m, int n, double *outData );

//In Matlab: belTrans(:,:,t) = crfTransExpC(bel(:,t), tmp(:)', TransMat);
void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	 int i, j, Q;
	 double *TransMat,*belTransData, *belInData, *belttp1, *transTotal, *tempBel, maxVal;
	 mxArray *tempDoubleMat, *tempDoubleMat2;
	 if ((nlhs> 1) | (nrhs != 3))
	{
		mexErrMsgTxt("The number of input should be 3 and output arguments should be 1");
	}
	 Q = mxGetM(prhs[2]);

	// Output 
	plhs[0] = mxCreateDoubleMatrix(Q, Q, mxREAL);
	belTransData = mxGetPr(plhs[0]);

	// Input
	belInData = mxGetPr(prhs[0]);
	tempBel = mxGetPr(prhs[1]);
	TransMat = mxGetPr(prhs[2]);

	tempDoubleMat = mxCreateDoubleMatrix(Q,Q, mxREAL);
	belttp1 = mxGetPr(tempDoubleMat);

	tempDoubleMat2 = mxCreateDoubleMatrix(1,1, mxREAL);
	transTotal = mxGetPr(tempDoubleMat2);

	for(i=0;i<Q;i++)
	{
		for(j=0;j<Q;j++)
		{
			belttp1[j*Q+i] = belInData[i] + tempBel[j] + TransMat[j*Q+i];
			maxVal = max(maxVal, belttp1[i*Q+j]);
		}
	}
	logSumWithMax(belttp1, maxVal, Q, Q, transTotal);

	for(i=0;i<Q;i++)
	{
		for(j=0;j<Q;j++)
		{
			belTransData[i*Q+j] =exp(belttp1[i*Q+j] - *transTotal);
		}
	}
}


void logSumWithMax(double *inData, double maxVal, int m, int n, double *outData)
{
	 int i, j;
	 double tempVal;

	tempVal = 0;
	for (j = 0; j < m*n; j++)
	{
		tempVal +=exp(inData[j] - maxVal);
	}
	outData[0]=log(tempVal) + maxVal;
}
             
