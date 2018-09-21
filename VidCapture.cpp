#include "stdafx.h"
#include <iostream>
#include <vector>
#include <stdio.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// RobotComms includes
#using <system.dll>
#using <mscorlib.dll>
using namespace System;
using namespace System::IO::Ports;


using namespace cv;
using namespace std;

string window_name = "Original Video | q to quit";

// Stores HSV colour details for particular named block
// Also stores signal to be output to robot
// Stores x and y positions of centroid for a locating block
class Block
{
public:
	// members
	// H - Hue, S - Saturation, V - Value
	int iLowH;
	int iHighH;
	int iLowS;
	int iHighS;
	int iLowV;
	int iHighV;
	unsigned char signal; // bit signal to be sent to serial port
	string name;
	double xPos;
	double yPos;
	bool Found;

	// constructor and destructor
	Block() {};
	~Block() {};
};

// Global Variables
// Required for tracker bar debugging to refine the HSV values for block detection
vector <Block> Blocks;
Mat frame;

// Sets HSV values for block depending on colour.
// Also sets a name, and specific signal for use with robot / serial port
void inialiseBlocks(vector <Block> &bvec)
{
	Block br;
	br.name = "Red";
	br.iLowH = 170;
	br.iHighH = 179;
	br.iLowS = 150;
	br.iHighS = 255;
	br.iLowV = 60;
	br.iHighV = 255;
	br.signal = 0b00000001;
	bvec.push_back(br);

	Block by;
	by.name = "Yellow";
	by.iLowH = 4;
	by.iHighH = 30;
	by.iLowS = 90;
	by.iHighS = 255;
	by.iLowV = 60;
	by.iHighV = 255;
	by.signal = 0b00000010;
	bvec.push_back(by);

	// Went with Green colour instead of White
	// Difficult to determine white from the grey background
	// Had to use a fine range for saturation (0-13)
	// All other blocks had their own Hue ranges
	Block bg;
	bg.name = "Green";
	bg.iLowH = 10; //50
	bg.iHighH = 160; // 80 // 90-130 seems to be good range for green
	bg.iLowS = 170; //194 //150 default //190-195 is good range for green
	bg.iHighS = 255;
	bg.iLowV = 60;
	bg.iHighV = 255;
	bg.signal = 0b00000100;
	bvec.push_back(bg);
}


// Standard Image processing functions
// Converts BGR to HSV, and identifies qualifying pixels in image.
// Dilate/Erode functions coalesce the pixels in the image to form blobs
void ProcessImage(Mat& src, Block& bproc, Mat& dst)
{
	Mat imgHSV;

	//Convert the captured frame from BGR to HSV
	cvtColor(src, imgHSV, COLOR_BGR2HSV);

	// is this pixel the expected colour? 
	inRange(imgHSV, Scalar(bproc.iLowH, bproc.iLowS, bproc.iLowV), Scalar(bproc.iHighH, bproc.iHighS, bproc.iHighV), dst); //Threshold the image

																														   //morphological opening (removes small objects from the foreground)
	erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	//morphological closing (removes small holes from the foreground)
	dilate(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(dst, dst, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
}



// find contours in proc and draw to dst
// Draw an outline around the largest blob
void ObjectBounding(Mat& proc, Mat& dst)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(proc, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects and circles
	vector<Point> contour_poly;
	Rect boundRect;
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());

	/// Draw polygonal contour + bonding rects
	dst = Mat::zeros(proc.size(), CV_8UC3);

	// Find the largest contour
	int largestIndex = 0;
	int largestContour = 0;
	int secondLargestIndex = 0;
	int secondLargestContour = 0;
	for (int i = 0; i< contours.size(); i++)
	{
		if (contours[i].size() > largestContour)
		{
			secondLargestContour = largestContour;
			secondLargestIndex = largestIndex;
			largestContour = contours[i].size();
			largestIndex = i;
		}
		else if (contours[i].size() > secondLargestContour)
		{
			secondLargestContour = contours[i].size();
			secondLargestIndex = i;
		}
	}
	Scalar color = Scalar(0, 0, 255);
	// Draw the largest contour in dst
	approxPolyDP(Mat(contours[largestIndex]), contour_poly, 3, true);
	boundRect = boundingRect(Mat(contour_poly));

	drawContours(dst, contours, largestIndex, color, 5, 8);
	rectangle(dst, boundRect, color, 10, 8, 0);
}

// Checks the count of white pixels against the threshold
// If the count is higher then the block has been found
bool FoundBlock(Mat &vidsrc, Block &b)
{
	b.Found = false;
	Mat imgproc;
	ProcessImage(vidsrc, b, imgproc);
	//----//imshow("Processed Window", imgproc);
	imshow(b.name, imgproc); // static imaging - debugging

							 //Calculate the moments of the thresholded image
							 // if moments area is not large enough, return false
	Moments oMoments = moments(imgproc);
	double dArea = oMoments.m00;
	double w = imgproc.size().width;
	double h = imgproc.size().height;

	double x = w*h;
	// 70% of the area must be white pixels
	double threshold = x*.7;

	if (dArea < threshold)
	{
		return false;
	}

	// Once found draw a bouding box around block
	// Calculate centroid for block positioning
	b.Found = true;
	Mat contour;
	ObjectBounding(imgproc, contour);
	vidsrc = vidsrc + contour;
	imshow(window_name, vidsrc);
	b.xPos = oMoments.m10 / oMoments.m00;
	b.yPos = oMoments.m01 / oMoments.m00;
	return true;

}

// Open a COM port and send appropriate bit for the found block
void SendSignal(Block& b)
{
	cli::array<unsigned char>^ texBufArray = gcnew cli::array<unsigned char>(1);
	int baudRate = 9600;
	// robot interpreter box settings
	SerialPort^ robot_int; // signal
	robot_int = gcnew SerialPort("COM4", baudRate); // signal
	// open port
	try
	{
		// Open port to robot interpreter box
		robot_int->Open(); // signal

		// Set number to send to the port
		texBufArray[0] = b.signal;

		// Write number to the port
		robot_int->Write(texBufArray, 0, 1); // signal

		// close port to robot interpreter box
		robot_int->Close(); // signal
	}
	catch (IO::IOException^ e)
	{
		Console::WriteLine(e->GetType()->Name + ": Port is not ready");
	}
	cout << "Signal = " << (int)b.signal << "  " << b.name << "   x = " << b.xPos << "   y = " << b.yPos << endl;
}

// Debugging - shows a new window with tracker bars to determine HSV values
// Was most useful for testing the white block against grey background
const string trackbarWindowName = "Trackbars";
//This function gets called whenever a trackbar position is changed
void on_trackbar(int, void*)
{
	// Recalculates and Redraws in white window and colour image
	FoundBlock(frame, Blocks[2]); // [2] = green block
}

void createTrackbars(Block& b)
{
	//create window for trackbars
	namedWindow(trackbarWindowName, 0);
	//create trackbars and insert them into window
	//3 parameters are: the address of the variable that is changing when the trackbar is moved
	//the max value the trackbar can move 
	//and the function that is called whenever the trackbar is moved(eg. on_trackbar)
	//                                  ---->    ---->     ---->      
	createTrackbar("H_MIN", trackbarWindowName, &b.iLowH, b.iHighH, on_trackbar);
	createTrackbar("H_MAX", trackbarWindowName, &b.iHighH, b.iHighH, on_trackbar);
	createTrackbar("S_MIN", trackbarWindowName, &b.iLowS, b.iHighS, on_trackbar);
	createTrackbar("S_MAX", trackbarWindowName, &b.iHighS, b.iHighS, on_trackbar);
	createTrackbar("V_MIN", trackbarWindowName, &b.iLowV, b.iHighV, on_trackbar);
	createTrackbar("V_MAX", trackbarWindowName, &b.iHighV, b.iHighV, on_trackbar);
}


// Program entry point
int main()
{
	Block* LowestBlock;
	//----//namedWindow("Processed Window", WINDOW_NORMAL);
	inialiseBlocks(Blocks);
	namedWindow("Red", WINDOW_NORMAL); // static imaging - debug
	namedWindow("Yellow", WINDOW_NORMAL);// static imaging - debug
	namedWindow("Green", WINDOW_NORMAL);// static imaging - debug

	VideoCapture capture(0); //try to open string, this will attempt to open it as a video file
	if (!capture.isOpened())
	{
		cerr << "Failed to open a video device or video file!\n" << endl;
		return 1;
	}
	cout << "press q to quit" << endl;
	//namedWindow(window_name, CV_WINDOW_KEEPRATIO); //resizable window;
	namedWindow(window_name, WINDOW_NORMAL);
	//vidcap
	for (;;)
	{
		capture.read(frame); //vidcap
							 //frame = imread("C:\\Users\\LockTop\\Desktop\\aaaa.jpg", CV_LOAD_IMAGE_COLOR); // static image testing - debugging
		if (frame.empty()) //vidcap
			break; //vidcap

		imshow(window_name, frame);
		char key = (char)waitKey(5); //delay N millis, usually long enough to display and capture input
		if (key == 'q')
			return 0;
		// cycle through blocks, if found the bool will be changed to true
		// find all 3 coloured blocks
		for (int i = 0; i < Blocks.size(); i++)
		{
			FoundBlock(frame, Blocks[i]);
		}
		LowestBlock = NULL;  // there is no current lowestblock set yet
							 // cycle through blocks, 
							 // if all are found, determine the block with the lowest y value
							 // and send the signal for that block
		for (int i = 0; i < Blocks.size(); i++)
		{
			if (!Blocks[i].Found) // if not all blocks have been found then keep going until all found
			{
				continue;
			}
			if (LowestBlock)
			{
				//if (LowestBlock->yPos < Blocks[i].yPos)
				if (LowestBlock->xPos > Blocks[i].xPos)
					LowestBlock = &Blocks[i];
			}
			else
			{
				LowestBlock = &Blocks[i];
			}
		}
		if (LowestBlock)
		{
			SendSignal(*LowestBlock);
		}
		//createTrackbars(Blocks[2]); //- trackbar function call for debugging
	}
	waitKey(0);
	return 0;
}