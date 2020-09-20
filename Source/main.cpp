/*****************************************************************************/
/*
* @file main.cpp
*
* @desc sep2dfilt example
*
* @author  Sharon Golan
*
/*****************************************************************************/

#include <stdio.h>
#include <iostream>
#include <chrono>
#include "core/include/opencv2/core.hpp"
#include "highgui/include/opencv2/highgui.hpp"
#include "imgcodecs/include/opencv2/imgcodecs.hpp"
#include "imgproc/include/opencv2/imgproc.hpp"

using namespace cv;


void ExitPrompt(int aReturnCode)
{
	printf("\r\nPRESS ANY KEY TO EXIT...");
	getchar();
	exit( aReturnCode );
}


Mat	ImgIn ;		// input image
Mat	ImgOut;		// input image


int main()
{
	double Score;
	char strCudaLaunchBlocking[8];

	printf("### sep2dfilt example (filter coeff calculation & applying the filter  ###\r\n");
	std::cout << cv::getBuildInformation() << std::endl;
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

	float coeffs[5][5] = { {  4,  14,  22,  14,  4 },	//   58
						   { 14,  61, 101,  61, 14 },	//  251
						   { 22, 101, 164, 101, 22 },	//  410
						   { 14,  61, 101,  61, 14 },	//  251
						   {  4,  14, 22,   14,  4 } };	//   58
						                                //  ----
														// 1028
	//Mat kernel = (1.0f/1028.0f)*Mat(5, 5, CV_32F, coeffs);
	Mat kernel = (1.0/25.0)*Mat::ones(5,5, CV_32F);
	float m00 = kernel.at<float>(0);

	//Mat ImgOut= Mat::zeros(ImgIn.rows, ImgIn.cols, CV_32F);
	/// Initialize arguments for the filter
  	Point anchor = Point( -1, -1 );
  	int delta    = 0;
  	int ddepth   = -1;
	ImgIn.convertTo(ImgIn, CV_32F);
	filter2D(ImgIn, ImgOut, CV_32F, kernel, anchor, delta);
	normalize(ImgIn, ImgIn, 0, 1, cv::NORM_MINMAX);
	imshow("input", ImgIn);
	normalize(ImgOut, ImgOut, 0, 1, cv::NORM_MINMAX);
	imshow("filter2D result", ImgOut);

		
	printf("\r\nDone\r\n");
	waitKey(0);

	return 0;
}

