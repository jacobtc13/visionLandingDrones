#include <iostream>
#include <vector>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

// Generic
string window_name = "Original Video | q to quit";
const string trackbarWindowName = "Trackbars";
Mat src, src_gray, src_hsv;
Mat dst, detected_edges;

// Canny
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

// Sobel
Mat grad;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;
int c;

// Shi Tomasi
int maxCorners = 23;
int maxTrackbar = 100;
RNG rng(12345);

///////////////
/// Camera  ///
///////////////



//////////////////
/// Debugging  ///
//////////////////

// This function gets called whenever a trackbar position is changed
void on_trackbar(int, void*)
{
	// FIXME: Potentially delete this function
	// Recalculates and Redraws in white window and colour image
	//FoundBlock(frame, Blocks[2]); // [2] = green block
}

// Create trackbars for debugging
void createTrackbars(Block& b)
{
	//create window for trackbars
	namedWindow(trackbarWindowName, 0);

	//create trackbars and insert them into window FIXME: Possibly remove functions here
	createTrackbar("H_MIN", trackbarWindowName, &b.iLowH, b.iHighH, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &b.iHighH, b.iHighH, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &b.iLowS, b.iHighS, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &b.iHighS, b.iHighS, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &b.iLowV, b.iHighV, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &b.iHighV, b.iHighV, on_trackbar);
}

//////////////////////////
/// Feature Detection  ///
//////////////////////////

void MserFindFeatures()
{
    Ptr<MSER> ms = MSER::create();
    vector<vector<Point> > regions;
    vector<cv::Rect> mser_bbox;
    ms->detectRegions(img, regions, mser_bbox);
    
    for (int i = 0; i < regions.size(); i++)
    {
        rectangle(img, mser_bbox[i], CV_RGB(0, 255, 0));  
    }
    
    imshow("mser", img);
}

/// Example found at http://answers.opencv.org/question/4260/how-to-use-brisk-in-opencv/
void BriskFindFeatures()
{
	const char * PimA="box.png";   // object
   	const char * PimB="box_in_scene.png"; // image

   	cv::Mat GrayA =cv::imread(PimA);
   	cv::Mat GrayB =cv::imread(PimB);
   	std::vector<cv::KeyPoint> keypointsA, keypointsB;
   	cv::Mat descriptorsA, descriptorsB;
	//set brisk parameters

   	int Threshl=60;
   	int Octaves=4; (pyramid layer) from which the keypoint has been extracted
   	float PatternScales=1.0f;
	//declare a variable BRISKD of the type cv::BRISK

   	cv::BRISK  BRISKD(Threshl,Octaves,PatternScales);//initialize algoritm
   	BRISKD.create("Feature2D.BRISK");

   	BRISKD.detect(GrayA, keypointsA);
   	BRISKD.compute(GrayA, keypointsA,descriptorsA);

   	BRISKD.detect(GrayB, keypointsB);
   	BRISKD.compute(GrayB, keypointsB,descriptorsB);
}

/// Example found at http://answers.opencv.org/question/68547/opencv-30-fast-corner-detection/
void FastFindFeatures()
{
	vector<KeyPoint> keypointsD;
	Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
	vector<Mat> descriptor;

	detector->detect(src,keypointsD,Mat());
	drawKeypoints(src, keypointsD, src);
	imshow("keypoints",src);
}

void OrbFindFeatures()
{
	// Setup Orb features
	int MAX_FEATURES = 500;
	std::vector<KeyPoint> keypoints1;
  	Mat descriptors1;
   
  	// Detect ORB features and compute descriptors.
  	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  	orb->detectAndCompute(src, Mat(), keypoints1, descriptors1);
	drawKeypoints(src, keypoints1, src);
	imshow("keypoints",src);
}

////////////////////////
/// Corner Detection ///
////////////////////////

void HarrisCornerDetection()
{
	/// Set parameters
	Mat dst_norm, dst_norm_scaled;
  	dst = Mat::zeros( src.size(), CV_32FC1 );
  	int blockSize = 2;
  	int apertureSize = 3;
  	double k = 0.04;

	/// Perform Harris Corner Detection
  	cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  	convertScaleAbs( dst_norm, dst_norm_scaled );

	/// Draw circles where corners are
  	for( int j = 0; j < dst_norm.rows ; j++ )
    { 
		for( int i = 0; i < dst_norm.cols; i++ )
       	{
           	if( (int) dst_norm.at<float>(j,i) > thresh )
           	{
          		circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
           	}
       	}
    }

	/// Display image
  	namedWindow( corners_window, WINDOW_AUTOSIZE );
  	imshow( corners_window, dst_norm_scaled );
}

void ShiTomasiCornerDetection()
{
	if( maxCorners < 1 ) 
	{ 
		maxCorners = 1; 
	}

	/// Create vector for corners
  	vector<Point2f> corners;
  	double qualityLevel = 0.01;
  	double minDistance = 10;
  	int blockSize = 3;
  	bool useHarrisDetector = false;
  	double k = 0.04;
  	Mat copy;
  	copy = src.clone();
  	goodFeaturesToTrack( src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
  	int r = 4;

	/// Draw on image where corners are
  	for( size_t i = 0; i < corners.size(); i++ )
    { 
		circle( copy, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
	}

	/// Display image
  	namedWindow( source_window, WINDOW_AUTOSIZE );
  	imshow( source_window, copy );
}

//////////////////////
/// Edge Detection ///
//////////////////////

void LaplacianEdgeDetection()
{
 	/// Remove noise by blurring with a Gaussian filter
  	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  	/// Convert the image to grayscale
  	cvtColor( src, src_gray, CV_BGR2GRAY );

  	/// Create window
  	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  	/// Apply Laplace function
  	Mat abs_dst;
  	Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( dst, abs_dst );

  	/// Show what you got
  	imshow( window_name, abs_dst );
}

void SobelEdgeDetection()
{
	/// Remove noise by blurring with a Gaussian filter
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

  	/// Convert it to gray
  	cvtColor( src, src_gray, CV_BGR2GRAY );

  	/// Create window
  	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  	/// Generate grad_x and grad_y
  	Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y;

  	/// Gradient X
  	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_x, abs_grad_x );

  	/// Gradient Y
  	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_y, abs_grad_y );

  	/// Total Gradient (approximate)
  	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	imshow( window_name, grad );
}

void CannyEdgeDetection()
{
	/// Reduce noise with a kernel 3x3
  	blur( src_gray, detected_edges, Size(3,3) );

  	/// Canny detector
  	Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  	/// Using Canny's output as a mask, we display our result
  	dst = Scalar::all(0);

  	src.copyTo( dst, detected_edges);
	imshow( window_name, dst );
}

////////////////////////
/// Other Detection  ///
////////////////////////

/// Example at https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
void HoughCircles()
{
	/// Convert to HSV colour space
	cvtColor(src, src_hsv, COLOR_BGR2HSV);

	/// Create vector to store circles
	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
 	HoughCircles( src_hsv, circles, CV_HOUGH_GRADIENT, 1, src_hsv.rows/8, 200, 100, 0, 0 );

  	/// Draw the circles detected
  	for( size_t i = 0; i < circles.size(); i++ )
  	{
    	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      	int radius = cvRound(circles[i][2]);
      	// circle center
      	circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      	// circle outline
      	circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
   	}
}



// Program entry point
int main()
{
	/// Load an image 	TODO: Change to camera load function 
  	src = imread();

  	if( !src.data )
  	{ return -1; }

  	/// Create a matrix of the same type and size as src (for dst) - for debug
  	dst.create( src.size(), src.type() );

  	/// Convert the image to grayscale or HSV
  	//cvtColor( src, src_gray, CV_BGR2GRAY );

  	/// Create a window for debug
  	//namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  	/// Create a Trackbar for user to enter threshold for debug
  	//createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  	/// Show the image
  	CannyThreshold(0, 0);

  	/// Wait until user exit program by pressing a key
  	waitKey(0);
	  
	return 0;
}