#include "mex.h"
#define max(x1,x2) ((x1) > (x2))? (x1):(x2)

void mexFunction(int nlhs, mxArray *plhs[], 
    int nrhs, const mxArray *prhs[])
{
	 int i, j, Q, T, T1, T2, N;
	 double *inData1,*inData2,*outData, *tempMultMat, *maxVal, tempVal;
	 mxArray *tempDoubleMat, *maxDoubleMat;
	 if ((nlhs> 1) | (nrhs != 2))
	{
		mexErrMsgTxt("The number of input should be 2 and output arguments should be 1");
	}
	 // Matrix multiplication: QxT * TxN = QxN
	 Q = mxGetM(prhs[0]);
	 T1 = mxGetN(prhs[0]);
	 T2 = mxGetM(prhs[1]);
	 N = mxGetN(prhs[1]);

	if (T1!=T2)
	{
		mexErrMsgTxt("Matrix multiplication: QxT * TxN = QxN. Error matrix dimensions of T don't match.");
	}
	T = T1;
	 
	plhs[0] = mxCreateDoubleMatrix(Q, N, mxREAL);

	tempDoubleMat = mxCreateDoubleMatrix(Q*N,T, mxREAL);
	tempMultMat = mxGetPr(tempDoubleMat);

	maxDoubleMat = mxCreateDoubleMatrix(Q*N,1, mxREAL);
	maxVal = mxGetPr(maxDoubleMat);

	inData1 = mxGetPr(prhs[0]);
	inData2 = mxGetPr(prhs[1]);
	outData = mxGetPr(plhs[0]);
		 
//	maxVal =inData[0];
	
	// Multiply values (in log space this becomes sum)
	for (i = 0; i < Q*N; i++)
	{
		for (j = 0; j < T; j++)
		{
			tempMultMat[(i*T)+j] = inData1[j*Q+(i/N)] + inData2[(i%N)*T+j];

			if (j==0)
			{
				maxVal[i] = tempMultMat[(i*T)+j];
			}
			else
			{
				maxVal[i] = max(maxVal[i],tempMultMat[(i*T)+j]);
			}			
		}
	}

	// Sum values (in log space this becomes logsum)
	for (i = 0; i < Q*N; i++)
	{
		tempVal = 0;
		for (j = 0; j < T; j++)
		{
			if (mxIsInf(maxVal[i])!=true)
			{		
				tempVal +=exp(tempMultMat[(i*T)+j] - maxVal[i]);
			}
		}
		outData[((i%N)*Q)+(i/N)] = log(tempVal) + maxVal[i];
	}
}
             