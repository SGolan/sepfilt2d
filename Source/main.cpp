/*****************************************************************************
* @file   main.cpp
* @brief  separable 2D-filter example
*         https://bartwronski.com/2020/02/03/separate-your-filters-svd-and-low-rank-approximation-of-image-filters/
* @author Sharon G.
*****************************************************************************/

#include <iostream>
#include <iomanip>
#include <chrono>
#include "core/include/opencv2/core.hpp"
#include "highgui/include/opencv2/highgui.hpp"
#include "imgcodecs/include/opencv2/imgcodecs.hpp"
#include "imgproc/include/opencv2/imgproc.hpp"

using namespace cv;

#define CV_FLOAT_TYPE CV_64F	
#if CV_FLOAT_TYPE == CV_32F	
    #define FLOAT float
#else
    #define FLOAT double
#endif

enum {GAUSSIAN_MCR, GAUSSIAN_CLASSIC};
#define GAUSSIAN_FILTER GAUSSIAN_MCR



void ExitPrompt(int aReturnCode)
{
	printf("\r\nPRESS ANY KEY TO EXIT...");
	getchar();
	exit( aReturnCode );
}


Mat	ImgIn 		;		// input image
Mat	ImgOut		;		// output image : traditional 2D filter
Mat	ImgOutSep	;		// output image : 2-pass 1D filter


int main()
{
	double Score;
	char strCudaLaunchBlocking[8];

	printf("### sep2dfilt example (filter coeff calculation & applying the filter  ###\r\n");
	//std::cout << cv::getBuildInformation() << std::endl;
	// read input image
	printf("\r\nLoading input image\r\n");
	std::string strImgInFilename = "./data/Lenna_grey.png";
	ImgIn = imread(strImgInFilename);
	int type = ImgIn.type();
	if (ImgIn.empty())
	{
		printf("Failed to load input image\r\n");
		ExitPrompt(1);
	}
	// convert input image to 32-bit float
	ImgIn.convertTo(ImgIn, CV_FLOAT_TYPE);

	// initialize 2D-gaussian-kernel
#if GAUSSIAN_FILTER == GAUSSIAN_MCR
	FLOAT coeffs[25] = { 4,  14,  22,  14,  4,	   //   58
						14,  61, 101,  61, 14,	   //  251
						22, 101, 164, 101, 22,	   //  410
						14,  61, 101,  61, 14,	   //  251
						 4,  14, 22,   14,  4  };  //   58
						                           //  ----
												    // 1028
#else
	FLOAT coeffs[25] = { 1,   4,  7,   4,   1,	    //   17
						      4,  16, 26,  16,   4,	    //   66
						      7,  26, 41,  26,   7,	    //  107
						      4,  16, 26,  16,   4,	    //   66
						      1,   4,  7,   4,   1,  }; //  17
						                                //  ----
												        //  273
#endif
	Mat K = Mat(5, 5, CV_FLOAT_TYPE, coeffs);
	//kernel *= (1.0/1028.0);

	// print original 2D-filter
	std::cout << "2D filter:" << std::endl;
	std::cout << std::setw(4) << K << std::endl;

	// apply the 2D-filter
  	Point anchor =  Point(-1, -1);
  	int delta    =  0            ;
  	int ddepth   = -1            ;
	filter2D(ImgIn, ImgOut, CV_FLOAT_TYPE, K, anchor, delta);
	// normalize filtering result for display by imshow
	normalize(ImgIn, ImgIn, 0, 1, cv::NORM_MINMAX);
	// display the input & output images
	imshow("input", ImgIn);
	normalize(ImgOut, ImgOut, 0, 1, cv::NORM_MINMAX);
	imshow("filter2D result", ImgOut);

	// decomposing the 5x5 2D-filter into two 1x5 1D filters
	Mat Sv, U, Vt;
	SVDecomp(K, Sv, U, Vt, cv::SVD::FULL_UV);
	std::cout << "Sv:" << std::endl;
	std::cout <<  Sv << std::endl;
	std::cout << "U :" << std::endl;
	std::cout <<  U << std::endl;
	std::cout << "Vt:" << std::endl;
	std::cout <<  Vt << std::endl;
	Mat S = Mat::zeros(5, 5, CV_FLOAT_TYPE);
	S.at<FLOAT>(0, 0) = Sv.at<FLOAT>(0);
	S.at<FLOAT>(1, 1) = Sv.at<FLOAT>(1);
	S.at<FLOAT>(2, 2) = Sv.at<FLOAT>(2);
	S.at<FLOAT>(3, 3) = Sv.at<FLOAT>(3);
	S.at<FLOAT>(4, 4) = Sv.at<FLOAT>(4);
	Mat Ktag1  = U*S*Vt;
	std::cout << "Ktag SVD:" << std::endl;
	std::cout <<  Ktag1 << std::endl;

	// extract the principal component to be the 1D filter vector
	Mat Kv = sqrt(S.at<FLOAT>(0, 0))*U.col(0);
	std::cout << "Kv:" << std::endl;
	std::cout <<  Kv << std::endl;
	Mat Ktag2  = Kv*Kv.t();
	std::cout << "Ktag principal component:" << std::endl;
	std::cout <<  Ktag2 << std::endl;

	// apply the seperated 1D filter
	sepFilter2D	(ImgIn, ImgOutSep, CV_FLOAT_TYPE, Kv, Kv);
	normalize(ImgOutSep, ImgOutSep, 0, 1, cv::NORM_MINMAX);
	// display sep-filter result
	imshow("sepFilter2D result", ImgOutSep);

	
	printf("\r\nDone\r\n");
	waitKey(0);

	return 0;
}

