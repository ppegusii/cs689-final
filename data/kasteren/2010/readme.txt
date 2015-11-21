Thank you for downloading my coding framework. This document explains how to use the code and how it is structured. If you have any questions or suggestions please email me at: tim0306@gmail.com 

This code was used for the experiments described in: 

Human Activity Recognition from Wireless Sensor Network Data: Benchmark and Software
T.L.M. van Kasteren, G. Englebienne and Ben Krose
Activity Recognition in Pervasive Intelligent Environments
Atlantis Ambient and Pervasive Intelligence, Atlantis Press, 
Volume Editors: Liming Chen, Chris Nugent, Jit Biswas, Jesse Hoey, 2010.

Please refer to this book chapter when you include experiments on this coding framework in your publications.

Future datasets and publications can be found at: https://sites.google.com/site/tim0306/

Contents of dataset package: 
- Code for Naive Bayes, HMM, HSMM and CRF


Using the code in matlab:
----------------------------
Add all the directories and subdirectories in the zip file to you matlab path and run expScriptTimeLength.m or expScriptRepresentations.m to get the results of experiment 1 and 2 respectively.


Speeding up the code:
----------------------------
To make the CRF code run faster you will want to use the crfInferenceLog rather than crfInference in the crfLikelihoodDaysSplit.m file. To make crfInferenceLog work you have to compile the c code in the mex directory, by typing mex filename.c


Acknowledgement: 
----------------
There are some files in the tools directory that were taken from other packages. Some code for more speedy implementations was taken from Kevin Murphy's CRF toolbox http://www.cs.ubc.ca/~murphyk/Software/CRF/crf.html and the Hanso BFGS toolkit was used for the parameter estimations in CRF http://www.cs.nyu.edu/overton/software/hanso/.