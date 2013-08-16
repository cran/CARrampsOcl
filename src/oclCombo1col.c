// combo1colForR_d.c
// OpenCL version
// System incudes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// R includes
#include <R.h>
#include <Rmath.h>

// application includes
//#include "combo1colForR_d.h"

// OpenCL includes
 #include <CL/opencl.h>

// OpenCL kernel

const char* programSource =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void kronVectMult1colOnDevice(\n"
"  __global double *a,\n"
"  __global double *b,\n"
"  __global double *c,\n"
"  __global double *output,\n"
" const int na,\n"
" const int nb,\n"
" const int iter)\n"
"{\n"
"  double Csub = 0.0, currdiff, oldmean, oldsd ;  \n"
  "int N = na * nb,  acol,  bcol ; \n"
  "int idxtot = get_global_id(0); \n"
  "if( idxtot < N ) \n"
  "  { \n"
  "  int arow = idxtot/ nb; \n"
  "  int brow = idxtot % nb;\n"
  "  int idxtot = arow * nb + brow ; \n"
  "  oldmean = output[idxtot] ;\n"
  "  oldsd = output[idxtot+N] ;\n"
  "  double newmean ;\n"
  "    for( int k = 0; k < N; k++)\n"
  "       { \n"
  "         acol = k / nb ;\n"
  "         bcol = k % nb ;\n"
  "         Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] * c[k ] ;\n"
  "       } \n"
  "    currdiff = Csub - oldmean ; \n"
  "    newmean = oldmean + currdiff / (double) iter ; \n"
  "    output[idxtot] = newmean ; \n"
  "    output[idxtot+N] = oldsd + currdiff * (Csub - newmean) ; \n"
  "  } \n"
" }" 
;


void oclCombo1col( double *a, double *b, double *D, double *tausqy, 
      double* tausqphi, double *By, double *results,
      int *na1, int *nb1, int *nc1, int *F1) 
{

  int i, j, k, iter ;
  int na = na1[0],nb = nb1[0], nab = na1[0] * nb1[0], nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na*sizeof(double);
  size_t sizeb = nb * nb*sizeof(double);
  size_t sizec = nab * sizeof(double); // Changed from mat ver
  size_t sizer = 2 * nab * sizeof(double); // Changed from mat ver

  // allocate array on host
 
  Bphi = (double *)malloc(sizec);
  for (i = 0; i < nab; i++)
       Bphi[i] = 0.0f ;

    // Use this to check the output of each API call
    //     cl_int status;
    

    cl_int status;
    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------

    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;

    // Use clGetPlatformIDs() to retrieve the number of 
    // platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    // Allocate enough space for each platform
    platforms =
        (cl_platform_id*)malloc(
           numPlatforms*sizeof(cl_platform_id));
    
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms,
                NULL);

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    status = clGetDeviceIDs(
        platforms[0],
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);

    devices =
        (cl_device_id*)malloc(
            numDevices*sizeof(cl_device_id));


    status = clGetDeviceIDs(
        platforms[0],
        CL_DEVICE_TYPE_ALL,
        numDevices,
        devices,
        NULL);

    cl_context context = NULL;

    context = clCreateContext(
        NULL,
        numDevices,
        devices,
        NULL,
        NULL,
        &status);

    cl_command_queue cmdQueue;

    cmdQueue = clCreateCommandQueue(
        context,
        devices[0],
        0,
        &status);

    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //----------------------------------------------------- 

    cl_mem buffera;  // Input array on the device
    cl_mem bufferb;  // Input array on the device
    cl_mem bufferc;  // Input array on the device
    cl_mem bufferresult;  // Output array on the device


    buffera = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizea,
        NULL,
        &status);

    bufferb = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeb,
        NULL,
        &status);

    bufferc = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizec,
        NULL,
        &status);

    bufferresult = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizer,
        NULL,
        &status);

    status = clEnqueueWriteBuffer(
        cmdQueue,
        buffera,
        CL_TRUE,
        0,
        sizea,
        a,
        0,
        NULL,
        NULL);

    status = clEnqueueWriteBuffer(
        cmdQueue,
        bufferb,
        CL_TRUE,
        0,
        sizeb,
        b,
        0,
        NULL,
        NULL);

    status = clEnqueueWriteBuffer(
        cmdQueue,
        bufferc,
        CL_TRUE,
        0,
        sizec,
        Bphi,
        0,
        NULL,
        NULL);

    status = clEnqueueWriteBuffer(
        cmdQueue,
        bufferresult,
        CL_TRUE,
        0,
        sizer,
        results,
        0,
        NULL,
        NULL);
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        (const char**)&programSource,
        NULL,
        &status);

    status = clBuildProgram(
        program,
        numDevices,
        devices,
        NULL,
        NULL,
        NULL);

    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, "kronVectMult1colOnDevice", &status);

    status  = clSetKernelArg(
        kernel,
        0,
        sizeof(cl_mem),
        &buffera);
    status  |= clSetKernelArg(
        kernel,
        1,
        sizeof(cl_mem),
        &bufferb);
  /*  status  |= clSetKernelArg(
        kernel,
        2,
        sizeof(cl_mem),
        &bufferc);  */
    status  |= clSetKernelArg(
        kernel,
        3,
        sizeof(cl_mem),
        &bufferresult);
    status |= clSetKernelArg(
        kernel,
        4,
        sizeof(int),
       &na);
    status |= clSetKernelArg(
        kernel,
        5,
        sizeof(int),
       &nb);

  size_t globalWorkSize[1];
  globalWorkSize[0] = na * nb ;

// get R's RNG seed 
GetRNGstate();

for(i = 0; i < nc; i++)  // for each row in output
{

  for( j=0; j < nab; j++ )   // for each data element
  {
     neweigendenom = tausqy[i] ;
     for( k = 0; k < Fm1; k++)
        neweigendenom += D[j * Fm1 +k] * tausqphi[i * Fm1 + k ] ;
     normmean = tausqy[i] * By[j] / neweigendenom ;
     normstd = 1.0 / sqrt(neweigendenom) ;
     Bphi[j] = rnorm( normmean, normstd ) ;
  }
  
    status = clEnqueueWriteBuffer(
        cmdQueue,
        bufferc,
        CL_TRUE,
        0,
        sizec,
        Bphi,
        0,
        NULL,
        NULL);  

  iter = i + 1 ;

    status  = clSetKernelArg(
        kernel,
        2,
        sizeof(cl_mem),
        &bufferc);  
    status |= clSetKernelArg(
        kernel,
        6,
        sizeof(int),
       &iter);

   // new 07/26/13
           clEnqueueBarrier(cmdQueue);
   

  // do calculation on device:

    status = clEnqueueNDRangeKernel(
        cmdQueue,
        kernel,
        1,
        NULL,
        globalWorkSize,
        NULL,
        0,
        NULL,
        NULL);

   // new 07/26/13
       clEnqueueBarrier(cmdQueue);
   //
}

// done gen rand numbers; send seed state back to R
PutRNGstate(); 

  // Retrieve result from device 


    clEnqueueReadBuffer(
        cmdQueue,
        bufferresult,
        CL_TRUE,
        0,
        sizer,
        results,
        0,
        NULL,
        NULL);

  // clean up]

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(buffera);
    clReleaseMemObject(bufferb);
    clReleaseMemObject(bufferc);
    clReleaseMemObject(bufferresult);
    clReleaseContext(context);

    free(Bphi);
    free(platforms);
    free(devices);

}

