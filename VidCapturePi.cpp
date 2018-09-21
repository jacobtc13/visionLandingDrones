#include <iostream>
#include <vector>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;
using namespace std;

// Generic
Mat src, src_filt, dst;

// Feature detection
vector<KeyPoint> keypoints;
Mat descriptors;


///////////////
/// Camera  ///
///////////////



//////////////////
/// Debugging  ///
//////////////////

// This function gets called whenever a trackbar position is changed
void on_trackbar(int, void*)
{
	// Do nothing
}

// Create trackbars for debugging
void createTrackbars(Block& b)
{
	// Trackbar window name
	string trackbarWindowName = "Trackbars";

	// create window for trackbars
	namedWindow("Trackbars", 0);

	// create trackbars and insert them into window 
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

// API Reference: https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html#a136c6b29fbcdd783a1003bf429fa17ed
void MserFindFeatures()
{
	// Apply filtering/colourspace changes to src image
	//

	// Setup alg parameters
	vector<vector<Point> > regions;
    vector<Rect> mser_bbox;

	int = delta = 5;
	int = min_area = 60;
	int = max_area = 14400;
	double = max_variation = 0.25;
	double = min_diversity = .2;
	int = max_evolution = 200;
	double = area_threshold = 1.01;
	double = min_margin = 0.003;
	int = edge_blur_size = 5;

	// Initialise algorithm with parameters
	Ptr<MSER> ms = MSER::create(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);

	// Detect features
    ms->detectRegions(src, regions, mser_bbox);
    
	// Draw features on image
    for (int i = 0; i < regions.size(); i++)
    {
        rectangle(src, mser_bbox[i], CV_RGB(0, 255, 0));  
    }
    
	// Show image
    imshow("MSER Feature Detection", src);
}

/// Example found at http://answers.opencv.org/question/4260/how-to-use-brisk-in-opencv/
/// Example compares 2 different images
void BriskFindFeatures()
{
	// Apply filtering/colourspace changes to src image
	//

	// Load in image to be matched TODO: Replace with something or fix
   	const char * PimB="box_in_scene.png";
	Mat src_obj = imread(PimB);
   	
	// Setup alg parameters
	vector<KeyPoint> keypoints_obj;
   	Mat descriptors_obj;
   	int Thresh = 30; // originally 60
   	int Octaves = 3; // (pyramid layer) from which the keypoint has been extracted, originally 4
   	float PatternScales = 1.0f;

	// Initialise algorithm with parameters
	Ptr<BRISK> detector = BRISK::create(Thresh, Octaves, PatternScales);

	// Detect features
   	detector->detectAndCompute(src, Mat(), keypoints, descriptors);
   	detector->detectAndCompute(src_obj, Mat(), keypoints_obj, descriptors_obj);

	// Draw features on image
	//

	// Show image
	//
}

/// Example found at http://answers.opencv.org/question/68547/opencv-30-fast-corner-detection/
/// Class reference: https://docs.opencv.org/3.4/df/d74/classcv_1_1FastFeatureDetector.html
void FastFindFeatures()
{
	// Apply filtering/colourspace changes to src image
	//

	// Setup alg parameters
	int threshold = 10;
	bool nonmaxSuppression = true;
	int	type = FastFeatureDetector::TYPE_9_16; // Can be TYPE_7_12 and TYPE_5_8

	// Initialise algorithm with parameters
	Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(threshold, nonmaxSuppression, type);

	// Detect features
	fast->detect(src, keypoints, Mat());

	// Draw features on image
	drawKeypoints(src, keypoints, src);

	// Show image
	imshow("Fast Find Features",src);
}

/// Class Reference: 
void OrbFindFeatures()
{
	// Apply filtering/colourspace changes to src image
	//

	// Setup alg parameters
	int	nfeatures = 500;
	float scaleFactor = 1.2f;
	int	nlevels = 8;
	int	edgeThreshold = 31;
	int	firstLevel = 0;
	int	WTA_K = 2;
	int	scoreType = ORB::HARRIS_SCORE; // Can also be FAST_SCORE
	int	patchSize = 31;
	int	fastThreshold = 20; 
   
  	// Initialise algorithm with parameters
  	Ptr<ORB> orb = ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
  	
	// Detect features  
	orb->detectAndCompute(src, Mat(), keypoints, descriptors);

	// Draw features on image
	drawKeypoints(src, keypoints, src);

	// Show image
	imshow("ORB Find Features",src);
}

////////////////////////
/// Corner Detection ///
////////////////////////

void HarrisCornerDetection()
{
	// Apply filtering/colourspace changes to src image
	cvtColor( src, src_filt, CV_BGR2GRAY );

	// Setup alg parameters
	Mat dst_norm, dst_norm_scaled;
  	dst = Mat::zeros( src.size(), CV_32FC1 );
  	int blockSize = 2;
  	int apertureSize = 3;
  	double k = 0.04;

	// Detect features  
  	cornerHarris( src_filt, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  	convertScaleAbs( dst_norm, dst_norm_scaled );

	// Draw features on image
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

	// Show image
  	imshow( "Harris Corner Detection", dst_norm_scaled );
}

void ShiTomasiCornerDetection()
{
	// Apply filtering/colourspace changes to src image
	cvtColor( src, src_filt, CV_BGR2GRAY );

	// Safety check
	int maxCorners = 23;
	if( maxCorners < 1 ) 
	{ 
		maxCorners = 1; 
	}

	// Setup alg parameters
  	vector<Point2f> corners;
  	double qualityLevel = 0.01;
  	double minDistance = 10;
  	int blockSize = 3;
  	bool useHarrisDetector = false;
  	double k = 0.04;
	int maxTrackbar = 100;
	RNG rng(12345);

	// Detect features  
  	goodFeaturesToTrack( src_filt, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
  	int r = 4;

	// Draw features on image
  	for( size_t i = 0; i < corners.size(); i++ )
    { 
		circle( src_filt, corners[i], r, Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)), -1, 8, 0 ); 
	}

	// Show image
  	imshow( "Harris Corner Detection", src_filt );
}

//////////////////////
/// Edge Detection ///
//////////////////////

void LaplacianEdgeDetection()
{
	// Apply filtering/colourspace changes to src image
  	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  	cvtColor( src, src_filt, CV_BGR2GRAY );

	// Setup alg parameters
  	int kernel_size = 3;
  	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;
	Mat abs_dst;

  	// Detect features  
  	Laplacian( src_filt, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( dst, abs_dst );

  	// Show image
  	imshow( "Laplacian Edge Detection", abs_dst );
}

void SobelEdgeDetection()
{
	// Apply filtering/colourspace changes to src image
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
  	cvtColor( src, src_filt, CV_BGR2GRAY );

  	/// Create window
  	//namedWindow( "Sobel Edge Detection", CV_WINDOW_AUTOSIZE );

	// Setup alg parameters
	Mat grad;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	//int c;

  	/// Generate grad_x and grad_y
  	Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y;

  	/// Gradient X
  	//Scharr( src_filt, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  	Sobel( src_filt, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_x, abs_grad_x );

  	/// Gradient Y
  	//Scharr( src_filt, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  	Sobel( src_filt, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  	convertScaleAbs( grad_y, abs_grad_y );

  	/// Total Gradient (approximate)
  	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	// Show image
	imshow( "Sobel Edge Detection", grad );
}

void CannyEdgeDetection()
{
	// Apply filtering/colourspace changes to src image
	Mat detected_edges;
  	blur( src_filt, detected_edges, Size(3,3) );
	cvtColor( src, src_filt, CV_BGR2GRAY );

	// Setup alg parameters
	int edgeThresh = 1;
	int lowThreshold;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;

  	// Detect features 
  	Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  	// Using Canny's output as a mask, we display our result
  	dst = Scalar::all(0);

	// Show image
  	src.copyTo( dst, detected_edges);
	imshow( "Canny Edge Detection", dst );
}

////////////////////////
/// Other Detection  ///
////////////////////////

/// Example at https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html
// Class Reference: https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
void HoughCircles()
{
	// Apply filtering/colourspace changes to src image
	cvtColor(src, src_filt, COLOR_BGR2HSV);

	// Setup alg parameters
	vector<Vec3f> circles;
	int method = CV_HOUGH_GRADIENT;
	double dp = 1;
	double minDist = src_filt.rows/8;
	double param1 = 200;
	double param2 = 100;
	int minRadius = 0;
	int maxRadius = 0;

	// Detect features 
 	HoughCircles( src_filt, circles, method, dp, minDist, param1, param2, minRadius, maxRadius);

  	// Show image
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