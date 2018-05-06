/*
 *     Copyright (C) 2013-2018 Sumandeep Banerjee
 * 
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU Lesser General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU Lesser General Public License
 *     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/* 
 * File:   
 * Author: sumandeep
 * Email:  sumandeep.banerjee@gmail.com
*/

#include <iostream>
#include <string>
#include "FaceDetect.h"

using namespace std;
using namespace cv;

const String g_strFrontCascade = "haarcascade_frontalface_alt_tree.xml";  // frontal face detector
const String g_strLeftEyeCascade = "haarcascade_lefteye_2splits.xml";	  // left eye detector	
const String g_strRightEyeCascade = "haarcascade_righteye_2splits.xml";	  // right eye detector	
const String g_strNoseCascade = "haarcascade_mcs_nose.xml";	              // nose detector	
const String g_strMouthCascade = "haarcascade_mcs_mouth.xml";	          // mouth detector	

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
		3,	 // Min Neighbors
		0 | CV_HAAR_FIND_BIGGEST_OBJECT,
        Size(30, 30));
 
	// no face detected, return error
	if ( faces.size() <= 0 )
	{
		//cerr << "ERROR: Could not detect Frontal Face" << endl;
		return -1;
	}

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
