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
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

// OpenCL kernel

const char* programSource3 =
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"__kernel void kronVectMult1colOnDevice(\n"
"  __global double *a,\n"
"  __global double *b,\n"
"  __global double *b2,\n"
"  __global double *c,\n"
"  __global double *output,\n"
" const int na,\n"
" const int nb,\n"
" const int nb2,\n"
" const int iter)\n"
"{\n"
"  double Csub = 0.0, currdiff, oldmean, oldsd ;  \n"
  "int N = na * nb * nb2,  acol,  bcol, b2col ; \n"
  "int idxtot = get_global_id(0); \n"
  "if( idxtot < N ) \n"
  "  { \n"
  "  int arow = idxtot / (nb * nb2); \n"
  "  int brow = (idxtot / nb2) % nb;\n"
  "  int b2row = idxtot % nb2;\n"
  "  oldmean = output[idxtot] ;\n"
  "  oldsd = output[idxtot+N] ;\n"
  "  double newmean ;\n"
  "    for( int k = 0; k < N; k++)\n"
  "       { \n"
  "         acol = k /( nb * nb2 ) ;\n"
  "         bcol = k % (nb * nb2) / nb2 ;\n"
  "         b2col = k % nb2 ;\n"
  "         Csub += a[ arow * na +  acol ] * b[brow * nb +  bcol ] * b2[b2row * nb2 + b2col ] * c[k ] ;\n"
  "       } \n"
  "    currdiff = Csub - oldmean ; \n"
  "    newmean = oldmean + currdiff / (double) iter ; \n"
  "    output[idxtot] = newmean ; \n"
  "    output[idxtot+N] = oldsd + currdiff * (Csub - newmean) ; \n"
  "  } \n"
" }" 
;


void oclCombo1col3( double *a, double *b, double *b2, double *D, double *tausqy, 
      double* tausqphi, double *By, double *results,
      int *na1, int *nb1, int *nb21, int *nc1, int *F1) 
{

  int i, j, k, iter ;
  int na = na1[0],nb = nb1[0], nb2 = nb21[0], nab = na1[0] * nb1[0] * nb21[0], nc = nc1[0], F=F1[0];
  int Fm1 = F - 1 ;

  double *Bphi ;
  double neweigendenom, normmean, normstd ;

  size_t sizea = na * na * sizeof(double);
  size_t sizeb = nb * nb * sizeof(double);
  size_t sizeb2 = nb2 * nb2 * sizeof(double);
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

    if (status != CL_SUCCESS) {
//        printf( "Error getting platform id %d.\n", status );
  //      exit(status);
     }

    // Allocate enough space for each platform
    platforms =
        (cl_platform_id*)malloc(
           numPlatforms*sizeof(cl_platform_id));
    
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms,
                NULL);

    if (status != CL_SUCCESS) {
    //    printf( "Error getting platform id.\n" );
    //    exit(status);
     }

    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    status = clGetDeviceIDs(
        platforms[0],
        CL_DEVICE_TYPE_ALL,
        0,
        NULL,
        &numDevices);

/*
    if (status != CL_SUCCESS) {
        printf( "Error getting device id.\n" );
        exit(status);
     }
*/

    devices = (cl_device_id*)malloc(
            numDevices*sizeof(cl_device_id));


    status = clGetDeviceIDs(
        platforms[0],
        CL_DEVICE_TYPE_ALL,
        numDevices,
        devices,
        NULL);

/*
    if (status != CL_SUCCESS) {
        printf( "Error getting device id.\n" );
        exit(status);
     }
*/

    cl_context context = NULL;

    context = clCreateContext(
        NULL,
        numDevices,
        devices,
        NULL,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating context.\n" );
        exit(status);
     }
*/

    cl_command_queue cmdQueue;

    cmdQueue = clCreateCommandQueue(
        context,
        devices[0],
        0,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating command queue.\n" );
        exit(status);
     }
*/

    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //----------------------------------------------------- 

    cl_mem buffera;  // Input array on the device
    cl_mem bufferb;  // Input array on the device
    cl_mem bufferb2;  // Input array on the device
    cl_mem bufferc;  // Input array on the device
    cl_mem bufferresult;  // Output array on the device


    buffera = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizea,
        NULL,
        &status);

/*
    if (status != CL_SUCCESS) {
        printf( "Error creating buffera.\n" );
        exit(status);
     }
*/

    bufferb = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeb,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating bufferb.\n" );
        exit(status);
     }
*/

    bufferb2 = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizeb2,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating bufferb2.\n" );
        exit(status);
     }
*/

    bufferc = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        sizec,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating bufferc.\n" );
        exit(status);
     }
*/

    bufferresult = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizer,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating bufferresult.\n" );
        exit(status);
     }
*/

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

/*
    if (status != CL_SUCCESS) {
        printf( "Error writing buffera.\n" );
        exit(status);
     }
*/

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

/*
    if (status != CL_SUCCESS) {
        printf( "Error writing bufferb.\n" );
        exit(status);
     }
*/

    status = clEnqueueWriteBuffer(
        cmdQueue,
        bufferb2,
        CL_TRUE,
        0,
        sizeb2,
        b2,
        0,
        NULL,
        NULL);

/*
    if (status != CL_SUCCESS) {
        printf( "Error writing bufferb2.\n" );
        exit(status);
     }
*/

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
/*
    if (status != CL_SUCCESS) {
        printf( "Error writing bufferb.\n" );
        exit(status);
     }
*/

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
/*
    if (status != CL_SUCCESS) {
        printf( "Error writing bufferresult %d.\n", status );
        exit(status);
     }
*/


    cl_program program = clCreateProgramWithSource(
        context,
        1,
        (const char**)&programSource3,
        NULL,
        &status);
/*
    if (status != CL_SUCCESS) {
        printf( "Error creating program with source %d.\n", status );
        printf(" This is the new version.\n") ;
        
        exit(status);
     }
*/


    status = clBuildProgram(
        program,
        numDevices,
        devices,
        NULL,
        NULL,
        NULL);
/*
    if (status != CL_SUCCESS) {
        printf( "Error building program %d.\n", status );
	char buffer[10240];
	clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
               sizeof(buffer), buffer, NULL);
	printf("CL Compilation failed:\n%s", buffer);
        exit(status);
     }
*/


    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, "kronVectMult1colOnDevice", &status);

/*
    if (status != CL_SUCCESS) {
        printf( "Error creating kernel.\n" );
        exit(status);
     }
*/


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
    status  |= clSetKernelArg(
        kernel,
        2,
        sizeof(cl_mem),
        &bufferb2);
    status  |= clSetKernelArg(
        kernel,
        4,
        sizeof(cl_mem),
        &bufferresult);
    status |= clSetKernelArg(
        kernel,
        5,
        sizeof(int),
       &na);
    status |= clSetKernelArg(
        kernel,
        6,
        sizeof(int),
       &nb);
    status |= clSetKernelArg(
        kernel,
        7,
        sizeof(int),
       &nb2);
/*
    if (status != CL_SUCCESS) {
        printf( "Error setting first group of arguments.\n" );
        exit(status);
     }
*/

  size_t globalWorkSize[1];
  globalWorkSize[0] = na * nb * nb2 ;

// get R's RNG seed 
GetRNGstate();

for(i = 0; i < nc; i++)  // for each row in output
{
// printf("%d \n", i) ;

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

/*
    if (status != CL_SUCCESS) {
        printf( "Error writing bufferc.\n" );
        exit(status);
     }
*/

    iter = i + 1 ;

    status  = clSetKernelArg(
        kernel,
        3,
        sizeof(cl_mem),
        &bufferc);  
    status |= clSetKernelArg(
        kernel,
        8,
        sizeof(int),
       &iter);
/*
    if (status != CL_SUCCESS) {
        printf( "Error setting 2nd pair of kernel arguments.\n" );
        exit(status);
     }
*/

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

/*
    if (status != CL_SUCCESS) {
        printf( "Error on NDRangeKernel.\n" );
        exit(status);
     }
*/
    // new 07/26/13
            clEnqueueBarrier(cmdQueue);
    

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
/*
    if (status != CL_SUCCESS) {
        printf( "Error reading buffers.\n" );
        exit(status);
     }
*/

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

