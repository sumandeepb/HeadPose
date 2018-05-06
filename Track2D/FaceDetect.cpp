/*
 *  Confidential and Proprietary!
 *  All Rights Reserved by Perceptive Devices LLC. 2013
 *  Copyrights owned by Perceptive Devices LLC.
 *
 */

#include <iostream>
#include <string>
#include "FaceDetect.h"

using namespace std;
using namespace cv;

const String g_strFrontCascade = "haarcascade_frontalface_alt_tree.xml";  // frontal face detector
const String g_strLeftEyeCascade = "haarcascade_lefteye_2splits.xml";	  // left eye detector	
const String g_strRightEyeCascade = "haarcascade_righteye_2splits.xml";	  // right eye detector	
const String g_strNoseCascade = "haarcascade_mcs_nose.xml";	  // right eye detector	
const String g_strMouthCascade = "haarcascade_mcs_mouth.xml";	  // right eye detector	

// Detect Frontal Face. Assume input image to be 8-bit grayscale
int detectFrontalFace( const Mat& imageFrame, Rect& faceRect )
{
	static CascadeClassifier cascadeFace;
	static bool fCascadeLoaded = false;

	// Load FACE
	if( !fCascadeLoaded )
    {
        if ( !cascadeFace.load( g_strFrontCascade ) )
		{
			cerr << "ERROR: Could not load Frontal Face classifier cascade" << endl;
			return -1;
		}
		fCascadeLoaded = true;
    }

	//Mat grayImg;
	vector<Rect> faces;

	// Convert to Grayscale
	//cvtColor( imageFrame, grayImg, CV_BGR2GRAY );
	
	// Detect Faces in the Image
	cascadeFace.detectMultiScale ( imageFrame, faces,
        1.1, // Scale Factor
		2,	 // Min Neighbors  (Try 3 for better accuracy 
		0
	    //|CV_HAAR_DO_CANNY_PRUNING   
        |CV_HAAR_FIND_BIGGEST_OBJECT   // *** Reduces time by 50+% (for Haar)!!
        // |CV_HAAR_DO_ROUGH_SEARCH
        // |CV_HAAR_SCALE_IMAGE
 		,
        Size(30, 30));  // Originally 30  (Does not make a difference if looking for Biggest Object; else 50x50 is better)
 
	// no face detected, return error
	if ( faces.size() <= 0 )
	{
		//cerr << "ERROR: Could not detect Frontal Face" << endl;
		return -1;
	}

	// sort faces from largest to smallest
	// sort

	// return largest face
	faceRect = faces[0];

	return 0;
}	// End of detectFrontalFace()

// detect eyes within the face rectangle
int detectEyes( const Mat& imageFrame, Rect& leftEyeRect, Rect& rightEyeRect, Rect& noseRect )
{
	static CascadeClassifier cascadeLeftEye;
	static CascadeClassifier cascadeRightEye;
	static CascadeClassifier cascadeNose;
	static bool fCascadeLoaded = false;

	// Load FACE
	if( !fCascadeLoaded )
    {
        if ( !cascadeLeftEye.load( g_strLeftEyeCascade ) )
		{
			cerr << "ERROR: Could not load Left Eye classifier cascade" << endl;
			return -1;
		}
        if ( !cascadeRightEye.load( g_strRightEyeCascade ) )
		{
			cerr << "ERROR: Could not load Right Eye classifier cascade" << endl;
			return -1;
		}
        if ( !cascadeNose.load( g_strNoseCascade ) )
		{
			cerr << "ERROR: Could not load Nose classifier cascade" << endl;
			return -1;
		}
		fCascadeLoaded = true;
    }

	vector<Rect> leftEye;
	vector<Rect> rightEye;
	vector<Rect> nose;

	// Detect Faces in the Image
	cascadeLeftEye.detectMultiScale( imageFrame, leftEye, 1.1, 3, 0|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
	cascadeRightEye.detectMultiScale( imageFrame, rightEye, 1.1, 3, 0|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));
	cascadeNose.detectMultiScale( imageFrame, nose, 1.1, 3, 0|CV_HAAR_FIND_BIGGEST_OBJECT, Size(20, 20));

	if( leftEye.size() <= 0 || rightEye.size() <= 0 || nose.size() <= 0)
	{
		//cerr << "ERROR: Could not detect Eyes and Nose" << endl;
		return -1;
	}

	leftEyeRect = leftEye[0];
	rightEyeRect = rightEye[0];
	noseRect = nose[0];

	return 0;
}