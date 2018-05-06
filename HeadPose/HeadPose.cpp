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
#include <math.h>
#include "HeadPose.h"
#include "FaceDetect.h"

//#undef _DEBUG

using namespace std;
using namespace cv;

// initialization code
CHeadPoseEstimator::CHeadPoseEstimator()
	:m_BRISKDetector(),
	m_BFMatcher( NORM_HAMMING )
{
	// init default values
	m_cameraModel = Mat::eye(3, 3, CV_64F);
	m_cameraRot = Mat::eye(3, 3, CV_64F);
	m_cameraTran = Mat::zeros(3, 1, CV_64F);
	// convert rotation matrix to rotation vector
	Rodrigues( m_cameraRot, m_cameraRVec );

	m_fKeyFrameDetected = false;
}

CHeadPoseEstimator::~CHeadPoseEstimator()
{
}

// frame feed from video capture device
int CHeadPoseEstimator::addNewFrame ( const Mat& inputFrame )
{
	cvtColor( inputFrame, m_currFrame, CV_BGR2GRAY );
	//equalizeHist( m_currFrame, m_currFrame );

	// if key frame is yet to be detected
	if ( !m_fKeyFrameDetected )
	{
		// return error if key frame is not detected
		if ( -1 == detectFrontalFace ( m_currFrame, m_faceRect ) )
		{
			cerr << "ERROR: Key frame not detected" << endl;
			return -1;
		}

		// store current frame as key frame
		m_keyFrame = m_currFrame.clone();

		// create image mask for the face rectangle
		m_maskFrame.create( m_keyFrame.rows, m_keyFrame.cols , CV_8UC1 );

		m_maskFrame.setTo( 0 );
		rectangle( m_maskFrame, Point( m_faceRect.x, m_faceRect.y ), 
			Point( m_faceRect.x+m_faceRect.width, m_faceRect.y+m_faceRect.height ), 
			CV_RGB( 255, 255, 255 ) );
		floodFill( m_maskFrame, Point( m_faceRect.x + 1, m_faceRect.y + 1 ), CV_RGB( 255, 255, 255 ) );

#if _DEBUG
		imwrite( "facemask.png", m_maskFrame ); // output mask frame
#endif

		// detect eyes within the face rectangle
		if( -1 == detectEyes( m_keyFrame, m_leftEyeRect, m_rightEyeRect ) )
		{
			cerr << "ERROR: Eyes not detected" << endl;
			return -1;
		}

		// detect keypoints and extract keypoint descriptors in the keyframe within the face rectangle (m) 
		m_BRISKDetector( m_keyFrame, m_maskFrame, m_keyFrameKeyPoints, m_keyFrameDescriptors );

#if _DEBUG
		Mat	keyPointImg;
		drawKeypoints( m_keyFrame, m_keyFrameKeyPoints, keyPointImg ); // draw keypoints on keyframe
		imwrite( "keypoints.png", keyPointImg );
#endif

		// backproject the keypoints to 3D Model (M)
#if 0 //_DEBUG
		cout << "2D -> 3D Point Correspondence\n";
#endif
		for ( vector<KeyPoint>::iterator it = m_keyFrameKeyPoints.begin(); it != m_keyFrameKeyPoints.end(); it++ )
		{
			Point2f m = it->pt;
			Point3f M = model2Dto3D( m, 0 );

			// create 2D point list from keypoint coordinates
			m_keyFrame2DPoints.push_back( it->pt );
			// create 3D point list from 3D Model backprojection
			m_keyFrame3DPoints.push_back( M );
#if 0 //_DEBUG
			cout << "(" << m.x << ", " << m.y << ") -> (" << M.x << ", " << M.y << ", " << M.z << ")\n";
#endif
		}

#if _DEBUG
		Write3DPointsToObj( "test1.obj", m_keyFrame3DPoints );
#endif

		// initial camera intrinsic matrix
		m_cameraModel.at<double>(0, 0) = 1000.0; // fx = 1000mm
		m_cameraModel.at<double>(1, 1) = 1000.0; // fy = 1000mm
		m_cameraModel.at<double>(0, 2) = m_lastFrame.cols / 2; // cx = W/2
		m_cameraModel.at<double>(1, 2) = m_lastFrame.rows / 2; // cy = H/2

		// compute camera pose from 2D-3D point correspondence
		solvePnPRansac( m_keyFrame3DPoints, m_keyFrame2DPoints, m_cameraModel, noArray(), m_cameraRVec, m_cameraTran ); 

		// convert rotation vector to rotation matrix
		Rodrigues( m_cameraRVec, m_cameraRot );

		// 3X4 rigid transformation matrix
		Mat	A = Mat::eye( 3, 4, CV_64F );
		for( int row = 0; row < 3; row++ )
		{
			for( int col = 0; col < 3; col++ )
			{
				A.at<double>(row, col) = m_cameraRot.at<double>(row, col);
			}
			A.at<double>(row, 3) = m_cameraTran.at<double>(row, 0);
		}

		// 3X4 perspective projection matrix
		m_cameraPers = m_cameraModel * A;

#if _DEBUG
		// print camera model
		cout << "Camera Model\n";
		for( int row = 0; row < m_cameraModel.rows; row++ )
		{
			for( int col = 0; col < m_cameraModel.cols; col++ )
			{
				cout << m_cameraModel.at<double>(row, col) << " ";
			}
			cout << endl;
		}

		// print camera pose
		cout << "Keyframe Camera Pose\n";
		for( int row = 0; row < m_cameraRot.rows; row++ )
		{
			for( int col = 0; col < m_cameraRot.cols; col++ )
			{
				cout << m_cameraRot.at<double>(row, col) << " ";
			}
			cout << m_cameraTran.at<double>(row, 0) << endl;
		}

		// backproject the keypoints from 2D to 3D
		vector<Point3f> test3Dpoints;
		for( vector<KeyPoint>::iterator it = m_keyFrameKeyPoints.begin(); it != m_keyFrameKeyPoints.end(); it++ )
		{
			Point2f m = it->pt;
			Point3f M = project2Dto3D( m );

			// create 3D point list from 2D to 3D backprojection
			test3Dpoints.push_back( M );
		}

		Write3DPointsToObj( "test2.obj", test3Dpoints );
#endif

		// copy keyframe data into last frame
		m_lastFrame = m_keyFrame.clone();
		m_lastFrameKeyPoints.assign( m_keyFrameKeyPoints.begin(), m_keyFrameKeyPoints.end() );
		m_lastFrameDescriptors = m_keyFrameDescriptors.clone();
		m_lastFrame2DPoints.assign( m_keyFrame2DPoints.begin(), m_keyFrame2DPoints.end() );
		m_lastFrame3DPoints.assign( m_keyFrame3DPoints.begin(), m_keyFrame3DPoints.end() );
		
		// set key frame detection flag
		m_fKeyFrameDetected = true;
	} // end of if( !m_fKeyFrameDetected )
	else // process frames
	{
		// current frame data
		vector<KeyPoint> currFrameKeyPoints;	// keypoints
		Mat				 currFrameDescriptors;	// keypoint descriptors
		vector<Point2f>  currFrame2DPoints;		// keypoint 2D coordinates
		vector<Point3f>	 currFrame3DPoints;		// keypoints on 3D Model 

		// detect keypoints and extract descriptors
		m_BRISKDetector( m_currFrame, m_maskFrame, currFrameKeyPoints, currFrameDescriptors );

		// determine keypoint matches between current frame and last frame
		vector<DMatch>	matches;
		m_BFMatcher.match( m_lastFrameDescriptors, currFrameDescriptors, matches );

#if _DEBUG
		// draw keypoint matches
		Mat	matchImage;
		drawMatches( m_lastFrame, m_lastFrameKeyPoints, m_currFrame, currFrameKeyPoints, matches, matchImage );
		imwrite( "matching.png", matchImage );
#endif

		// create 2D-3D point correspondence from keypoint matches
		for( vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++ )
		{
			currFrame3DPoints.push_back( m_lastFrame3DPoints[ it->queryIdx ] );
			currFrame2DPoints.push_back( currFrameKeyPoints[ it->trainIdx ].pt );
		}

		// compute camera pose from 2D-3D point correspondence (m = sA[R|T]M)
		vector<int>	inliers;
		solvePnPRansac( currFrame3DPoints, currFrame2DPoints, m_cameraModel, noArray(), m_cameraRVec, m_cameraTran,	false, 100, 8.0, 100, inliers );
		
		// convert rotation vector to rotation matrix
		Rodrigues( m_cameraRVec, m_cameraRot );

		// 3X4 rigid transformation matrix
		Mat	A = Mat::eye( 3, 4, CV_64F );
		for( int row = 0; row < 3; row++ )
		{
			for( int col = 0; col < 3; col++ )
			{
				A.at<double>(row, col) = m_cameraRot.at<double>(row, col);
			}
			A.at<double>(row, 3) = m_cameraTran.at<double>(row, 0);
		}

		// 3X4 perspective projection matrix
		m_cameraPers = m_cameraModel * A;

#if _DEBUG
		// create inlier mask for RANSAC
		vector<char> inlierMask;
		inlierMask.assign( matches.size(), 0 );
		for( vector<int>::iterator it = inliers.begin(); it != inliers.end(); it++ )
		{
			inlierMask[ *it ] = 1;
		}

		// draw keypoint matches
		Mat	matchImage2;
		drawMatches( m_lastFrame, m_lastFrameKeyPoints, m_currFrame, currFrameKeyPoints, matches, matchImage2,
			Scalar::all(-1), Scalar::all(-1), inlierMask);
		imwrite( "matching2.png", matchImage2 );
#endif

		//// determine relative pose from chained transformation
		//Mat	P0 = Mat::eye( 4, 4, CV_64F );
		//Mat	P1 = Mat::eye( 4, 4, CV_64F );
		////Mat	P01 = Mat::eye( 4, 4, CV_64F );

		//for( int row = 0; row < 3; row++ )
		//{
		//	for( int col = 0; col < 3; col++ )
		//	{
		//		P0.at<double>(row, col) = m_cameraRot.at<double>(row, col);
		//		P1.at<double>(row, col) = cameraRot.at<double>(row, col);
		//	}
		//	P0.at<double>(row, 3) = m_cameraTran.at<double>(row, 0);
		//	P1.at<double>(row, 3) = cameraTran.at<double>(row, 0);
		//}

		//m_HeadPose = P1 * P0.inv();
		//cout << "Relative Camera Pose\n";
		//for( int row = 0; row < 4; row++ )
		//{
		//	for( int col = 0; col < 4; col++ )
		//	{
		//		cout << m_HeadPose.at<double>(row, col) << " ";
		//	}
		//	cout << endl;
		//}

#if _DEBUG
		// print camera pose
		cout << "Absolute Camera Pose\n";
		for( int row = 0; row < m_cameraRot.rows; row++ )
		{
			for( int col = 0; col < m_cameraRot.cols; col++ )
			{
				cout << m_cameraRot.at<double>(row, col) << " ";
			}
			cout << m_cameraTran.at<double>(row, 0) << endl;
		}
#endif

		// re-projection
		// check number of valid reprojected keypoints in current keyframe
		// check for valid reprojection
		int nValidCount = 0;
		for( vector<KeyPoint>::iterator it = currFrameKeyPoints.begin(); it != currFrameKeyPoints.end(); it++ )
		{
			Point2f m = it->pt;
			Point3f M = project2Dto3D( m );

			// valid 3D point
			if( Point3f(0.0, 0.0, 0.0) != M )
			{
				nValidCount++;
			}
		}
		
		// accept current frame as reference, only if more that 4 valid points have been detected
		if( nValidCount >= 4 )
		{
			// copy current frame to last frame
			m_lastFrame = m_currFrame.clone();

			m_lastFrameKeyPoints.clear();
			m_lastFrame2DPoints.clear();
			m_lastFrame3DPoints.clear();

			// check for valid reprojection
			for( vector<KeyPoint>::iterator it = currFrameKeyPoints.begin(); it != currFrameKeyPoints.end(); it++ )
			{
				Point2f m = it->pt;
				Point3f M = project2Dto3D( m );

				// valid 3D point
				if( Point3f(0.0, 0.0, 0.0) != M )
				{
					// add keypoint
					m_lastFrameKeyPoints.push_back( *it );

					// add 2D point
					m_lastFrame2DPoints.push_back( m );

					// add 3D point
					m_lastFrame3DPoints.push_back( M );
				}
			}

			// recompute descriptors only for the valid keypoints
			m_BRISKDetector( m_lastFrame, m_maskFrame, m_lastFrameKeyPoints, m_lastFrameDescriptors, true );
		}

	} // end of else process frames

	return 0;
} // end of addNewFrame

// get head pose estimation
int CHeadPoseEstimator::getHeadPose()
{
	return 0;
}

// draw head pose coordinate axes on input frame
int CHeadPoseEstimator::drawPose( Mat& inputFrame )
{
#if 1
	rectangle( inputFrame, Point( m_faceRect.x, m_faceRect.y ), 
		Point( m_faceRect.x+m_faceRect.width, m_faceRect.y+m_faceRect.height ), 
		CV_RGB(255, 0, 0) );

	rectangle( inputFrame, Point( m_leftEyeRect.x, m_leftEyeRect.y ), 
		Point( m_leftEyeRect.x+m_leftEyeRect.width, m_leftEyeRect.y+m_leftEyeRect.height ), 
		CV_RGB(255, 255, 0) );

	rectangle( inputFrame, Point( m_rightEyeRect.x, m_rightEyeRect.y ), 
		Point( m_rightEyeRect.x+m_rightEyeRect.width, m_rightEyeRect.y+m_rightEyeRect.height ), 
		CV_RGB(0, 255, 0) );
#endif

	// dimension of coordinate axes
	const float axisLength = 100.0f;
	const float ROOT2 = 1.41421356f;

	// define the coordinate axes
	vector<Point3f> axes3D;
	vector<Point2f> axes2D;
	axes3D.push_back( Point3f( 0.0, 0.0, 0.0 ) ); 
	axes3D.push_back( Point3f( axisLength / ROOT2, axisLength / ROOT2, 0.0 ) ); 
	axes3D.push_back( Point3f( -axisLength / ROOT2, axisLength / ROOT2, 0.0 ) ); 
	axes3D.push_back( Point3f( 0.0, 0.0, axisLength ) ); 
	
	// project 3D points to 2D image plane
	projectPoints( axes3D, m_cameraRVec, m_cameraTran, m_cameraModel, noArray(), axes2D);

	// draw the coordinate axes
	line( inputFrame, axes2D[0], axes2D[1], CV_RGB(0, 255, 0), 1 );
	line( inputFrame, axes2D[0], axes2D[2], CV_RGB(0, 0, 255), 1 );
	line( inputFrame, axes2D[0], axes2D[3], CV_RGB(255, 0, 0), 1 );

	return 0;
}

// keyframe points to 3D Model
Point3f CHeadPoseEstimator::model2Dto3D( const Point2f& m, int method )
{
	// assume cylindrical 3D Model. later maybe change to 3D face mesh
	// assume average face radius ~ 90mm, width ~ 128mm, height ~ 128mm
	Point3f M;

	const float ROOT2 = 1.41421356f;
	const float R = 90.0f;
	const float H = 128.0f;
	const float W = ROOT2 * R;
	float xscale = W / float( m_faceRect.width );
	float yscale = H / float( m_faceRect.height );

	float d = xscale * float( m.x - m_faceRect.x );

	// solve quadratic equation
	if( 0 == method )
	{
		float a = 1.0f;
		float b = ROOT2 * d - R;
		float c = 0.5f * (b * b - R * R);
		float disc = b * b - 4.0f * a * c;
		float x1 = (-b + sqrt( disc )) / (2.0f * a);
		float x2 = (-b - sqrt( disc )) / (2.0f * a);
		M.x = x1;
		M.y = sqrt( R * R - M.x * M.x );
	}
	else if( 1 == method )
	{
		float d1 = 1.0f - ROOT2 * d / R;
		float d2 = 0.5f * (1.0f - d1 * d1);
		float disc = 1.0f - 4.0f * d2 * d2;
		float q1 = 0.5f * (1.0f + sqrt( disc ));
		float q2 = 0.5f * (1.0f - sqrt( disc ));
		M.x = R * sqrt( q1 );
		M.y = R * sqrt( 1.0f - q1 );
	}
	else if( 2 == method )
	{
		float base = ROOT2 * R - d;
		float hyp = sqrt( d * d + base * base );
		float perp = d;
		double theta = atan( perp / base );
		double k = sqrt( 4 * R * R * sin( 0.5 * theta ) * sin( 0.5 * theta ) - d * d );
		M.x = R * base / hyp;
		M.y = R * d / hyp;
	}

	M.z = 0.5f * H - yscale * float( m.y - m_faceRect.y ); 
	return M;
}

// backproject 2D keypoints(m) to 3D Model(M)
Point3f CHeadPoseEstimator::project2Dto3D( const Point2f& m )
{
	Point3f M(0.0, 0.0, 0.0);

	const double ROOT2 = 1.41421356;
	const double R = 90.0;
	const double H = 128.0;
	const double W = ROOT2 * R;

	// rigid transformation matrix elements
	double a11 = m_cameraPers.at<double>(0, 0);
	double a12 = m_cameraPers.at<double>(0, 1);
	double a13 = m_cameraPers.at<double>(0, 2);
	double a14 = m_cameraPers.at<double>(0, 3);
	double a21 = m_cameraPers.at<double>(1, 0);
	double a22 = m_cameraPers.at<double>(1, 1);
	double a23 = m_cameraPers.at<double>(1, 2);
	double a24 = m_cameraPers.at<double>(1, 3);
	double a31 = m_cameraPers.at<double>(2, 0);
	double a32 = m_cameraPers.at<double>(2, 1);
	double a33 = m_cameraPers.at<double>(2, 2);
	double a34 = m_cameraPers.at<double>(2, 3);
	double u = (double)m.x;
	double v = (double)m.y;

	// compute equation of line b1x + b2y + b3 = 0
	double b1 = (a11 - u * a31) / (a13 - u * a33) - (a21 - v * a31) / (a23 - v * a33);
	double b2 = (a12 - u * a32) / (a13 - u * a33) - (a22 - v * a32) / (a23 - v * a33);
	double b3 = (a14 - u * a34) / (a13 - u * a33) - (a24 - v * a34) / (a23 - v * a33);

	// solve line with equation of circle x2 + y2 = R2
	// we get quadratic equation ax2 + bx + c = 0;
	double a = b2 * b2 + b1 * b1;
	double b = 2.0 * b1 * b3;
	double c = b3 * b3 - b2 * b2 * R * R;

	// solve quadratic equation
	double disc = b * b - 4.0 * a * c;
	double x1 = (-b + sqrt( disc )) / (2.0 * a);
	double x2 = (-b - sqrt( disc )) / (2.0 * a);

	double X = x1;	// accept +ve solution
	double Y = - (b1 * X + b3) / b2; //sqrt( R * R - X * X );
	double Z = - (X * (a11 - u * a31) / (a13 - u * a33) + Y * (a12 - u * a32) / (a13 - u * a33) + (a14 - u * a34) / (a13 - u * a33));

	if (X >= 0.0 && Y >= 0.0 && Z >= -0.5 * H && Z <= 0.5 * H)
	{
		// return 3D Model point coordinates
		M.x = (float)X;
		M.y = (float)Y;
		M.z = (float)Z;
	}

	return M;
}

// write 3D points to Obj File
int Write3DPointsToObj( const char strFileName[], vector<Point3f>& points3D )
{
	FILE *pFile;

	pFile = fopen( strFileName, "wb" );
	if( NULL == pFile )
	{
		return -1;
	}

	for( vector<Point3f>::iterator it = points3D.begin(); it != points3D.end(); it++ )
	{
		fprintf( pFile, "v %f %f %f\n", it->x, it->y, it->z );
	}

	fclose(pFile);

	return 0;
}
