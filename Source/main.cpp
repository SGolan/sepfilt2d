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

#define CV_FLOAT_TYPE CV_32F	
#if CV_FLOAT_TYPE == CV_32F	
    #define FLOAT float
#else
    #define FLOAT double
#endif


void ExitPrompt(int aReturnCode)
{
	printf("\r\nPRESS ANY KEY TO EXIT...");
	getchar();
	exit( aReturnCode );
}


Mat	ImgIn 	 , ImgInDisplay		;		// input image
Mat	ImgOut   , ImgOutDisplay	;		// output image : traditional 2D filter
Mat	ImgOutSep, ImgOutSepDisplay	;		// output image : 2-pass 1D filter


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
	FLOAT coeffs[25] = { 4,  14,  22,  14,  4,	   //   58
						14,  61, 101,  61, 14,	   //  251
						22, 101, 164, 101, 22,	   //  410
						14,  61, 101,  61, 14,	   //  251
						 4,  14, 22,   14,  4  };  //   58
						                           //  ----
												    // 1028
	Mat K = Mat(5, 5, CV_FLOAT_TYPE, coeffs);
	//kernel *= (1.0/1028.0);

	// print original 2D-filter
	std::cout << "2D filter:" << std::endl;
	std::cout << std::setw(4) << K << std::endl;

	// apply the 2D-filter
  	Point anchor =  Point(-1, -1);
  	int delta    =  0            ;
  	int ddepth   = -1            ;
	auto start1 = std::chrono::system_clock::now();
	filter2D(ImgIn, ImgOut, CV_FLOAT_TYPE, K, anchor, delta);
	auto end1 = std::chrono::system_clock::now();
	auto elapsed1 =  std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);		
	// normalize filtering result for display by imshow
	normalize(ImgIn, ImgInDisplay, 0, 1, cv::NORM_MINMAX);
	// display the input & output images
	imshow("input", ImgInDisplay);
	normalize(ImgOut, ImgOutDisplay, 0, 1, cv::NORM_MINMAX);
	imshow("filter2D result", ImgOutDisplay);

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
	auto start2 = std::chrono::system_clock::now();
	sepFilter2D	(ImgIn, ImgOutSep, CV_FLOAT_TYPE, Kv, Kv.t());
	auto end2 = std::chrono::system_clock::now();
	auto elapsed2 =  std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);

	normalize(ImgOutSep, ImgOutSepDisplay, 0, 1, cv::NORM_MINMAX);
	// display sep-filter result
	imshow("sepFilter2D result", ImgOutSepDisplay);

	// zoon-in  
	cv::Rect CropRect(256, 256, 31, 31);
	Mat ImgInDisplayCrop     = ImgInDisplay    (CropRect); resize(ImgInDisplayCrop    , ImgInDisplayCrop    , Size(), 16, 16);
	Mat ImgOutDisplayCrop    = ImgOutDisplay   (CropRect); resize(ImgOutDisplayCrop   , ImgOutDisplayCrop   , Size(), 16, 16);
	Mat ImgOutSepDisplayCrop = ImgOutSepDisplay(CropRect); resize(ImgOutSepDisplayCrop, ImgOutSepDisplayCrop, Size(), 16, 16);
	imshow("Cropped input"             , ImgInDisplayCrop    );
	imshow("Cropped filter2D result"   , ImgOutDisplayCrop   );
	imshow("Cropped sepFilter2D result", ImgOutSepDisplayCrop);

	// compare images
	double MaxRelativeDiff = 0, AvRelativeDiff = 0 ;
	for(int i=3; i<ImgOut.rows-3; i++)
	{
   		for(int j=3; j<ImgOut.cols-3; j++) 
		{
			FLOAT pixel1 = ImgOut.at<FLOAT>(i, j)   ;
			FLOAT pixel2 = ImgOutSep.at<FLOAT>(i, j);
			double RelativeDiff = abs((pixel1 - pixel2)/pixel1);
			MaxRelativeDiff = max(MaxRelativeDiff, RelativeDiff);
			AvRelativeDiff += RelativeDiff;
		}
 	}
	AvRelativeDiff /= static_cast<double>(ImgOut.rows*ImgOut.cols);
	// print max relative diff
	std::cout << "max relative diff     = " << MaxRelativeDiff << std::endl;
	std::cout << "average relative diff = " << AvRelativeDiff  << std::endl;
	
	// print time measurements
	std::cout << "filter2D time    : "  <<  std::chrono::duration<double>(elapsed1).count() << "[ms]" << std::endl;	
	std::cout << "sepFilter2D time : "  <<  std::chrono::duration<double>(elapsed2).count() << "[ms]" << std::endl;					

	
	std::cout << "\r\nDone\r\n";
	waitKey(0);

	return 0;
}

