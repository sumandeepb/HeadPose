#include <iostream>
#include <math.h>
#include "Track2D.h"
#include "FaceDetect.h"

using namespace std;
using namespace cv;

CTrack2D::CTrack2D()
{
	// set flag for first frame
	m_fIsFirstFrame = true;
	m_nFrameCount = 0;

	// init variables
	m_omega = 0.0;
	//m_zoom = 0.0;
	m_xvelocity = 0.0;
	m_yvelocity = 0.0;
	//m_scale = 0.0;
	//m_xdisplacement = 0.0;
	//m_ydisplacement = 0.0;

	m_affineMatrix = Mat::eye( 3, 3, CV_64FC1 );
}

int CTrack2D::trackNewFrame( cv::Mat &inputFrame )
{
	// if this is the first frame
	if( m_fIsFirstFrame )
	{
		// convert frame to grayscale
		cvtColor( inputFrame, m_currFrame, CV_BGR2GRAY );

		m_maskFrame.create( m_currFrame.rows, m_currFrame.cols , CV_8UC1 );

		// return error if frontal face is not detected
		if ( -1 == detectFrontalFace ( m_currFrame, m_faceRect ) )
		{
			cerr << "ERROR: Frontal Face not detected" << endl;
			return -1;
		}

#ifndef FACIAL_FEATURES
		shrinkRect( m_faceRect );
		convertRectToPoly( m_faceRect, m_facePoly );
		copyPoly( m_facePoly, m_newFacePoly );

		// create image mask for the face rectangle
		m_maskFrame.setTo( 0 );
		fillConvexPoly( m_maskFrame, m_newFacePoly, 4, CV_RGB( 255, 255, 255 ) );
#else
		// detect eyes within the face rectangle
		if( -1 == detectEyes( m_currFrame, m_leftEyeRect, m_rightEyeRect, m_noseRect ) )
		{
			cerr << "ERROR: Eyes not detected" << endl;
			return -1;
		}
		
		shrinkRect( m_faceRect );
		convertRectToPoly( m_faceRect, m_facePoly );
		copyPoly( m_facePoly, m_newFacePoly );

		shrinkRect( m_leftEyeRect );
		convertRectToPoly( m_leftEyeRect, m_leftEyePoly );
		copyPoly( m_leftEyePoly, m_newLeftEyePoly );

		shrinkRect( m_rightEyeRect );
		convertRectToPoly( m_rightEyeRect, m_rightEyePoly );
		copyPoly( m_rightEyePoly, m_newRightEyePoly );

		shrinkRect( m_noseRect );
		convertRectToPoly( m_noseRect, m_nosePoly );
		copyPoly( m_nosePoly, m_newNosePoly );

		// create mask for eyes and nose
		m_maskFrame.setTo( 0 );
		
		fillConvexPoly( m_maskFrame, m_newLeftEyePoly, 4, CV_RGB( 255, 255, 255 ) );
		fillConvexPoly( m_maskFrame, m_newRightEyePoly, 4, CV_RGB( 255, 255, 255 ) );
		fillConvexPoly( m_maskFrame, m_newNosePoly, 4, CV_RGB( 255, 255, 255 ) );
#endif

#if _DEBUG
		imwrite( "facemask.png", m_maskFrame ); // output mask frame
#endif

		// detect good features to track
		goodFeaturesToTrack( m_currFrame, m_featuresCurrent, 30, 0.01, 30, m_maskFrame); //calculate the features

		// reset transformation matrix
		m_affineMatrix = Mat::eye( 3, 3, CV_64FC1 );

		// reset first frame flag
		m_fIsFirstFrame = false;
		m_nFrameCount = 0;
	} // end of if( fIsFirstFrame ) 
	else // from second frame onwards
	{
		// save current frame
		m_prevFrame = m_currFrame.clone();
		m_featuresPrevious = move( m_featuresCurrent );

		// convert frame to grayscale
		cvtColor( inputFrame, m_currFrame, CV_BGR2GRAY );

		// detect good features to track
		goodFeaturesToTrack( m_currFrame, m_featuresCurrent, 30, 0.01, 30, m_maskFrame ); //calculate the features

		// compute optical flow
		//vector<Mat>	pyramid;
		//int nLevels = buildOpticalFlowPyramid();
		calcOpticalFlowPyrLK( m_prevFrame, m_currFrame, m_featuresPrevious, m_featuresNextPos, m_featuresFound, m_flowErr );

		// compute transformation
		Mat affine = estimateRigidTransform( m_featuresPrevious, m_featuresNextPos, false );
		
#if _DEBUG
		// print camera model
		cout << "Relative Affine Transformation\n";
		for( int row = 0; row < affine.rows; row++ )
		{
			for( int col = 0; col < affine.cols; col++ )
			{
				cout << affine.at<double>(row, col) << " ";
			}
			cout << endl;
		}
#endif

		m_omega = asin( affine.at<double>(1, 0) );
		//m_zoom =  affine.at<double>(0, 0) / cos (m_omega);
		m_xvelocity = affine.at<double>(0, 2);
		m_yvelocity = affine.at<double>(1, 2);

		Mat relAffine = Mat::eye( 3, 3, CV_64FC1 );
		for( int row = 0; row < affine.rows; row++ )
		{
			for( int col = 0; col < affine.cols; col++ )
			{
				relAffine.at<double>(row, col) = affine.at<double>(row, col);
			}
		}

		m_affineMatrix = relAffine * m_affineMatrix;
		
#ifndef FACIAL_FEATURES
		// update 
		transformAffine( m_facePoly, m_newFacePoly );

		// create mask for face
		m_maskFrame.setTo( 0 );
		fillConvexPoly( m_maskFrame, m_newFacePoly, 4, CV_RGB( 255, 255, 255 ) );
#else
		// update 
		transformAffine( m_facePoly, m_newFacePoly );

		// update 
		transformAffine( m_leftEyePoly, m_newLeftEyePoly );
		transformAffine( m_rightEyePoly, m_newRightEyePoly );
		transformAffine( m_nosePoly, m_newNosePoly );
		
		// create mask for eyes and nose
		m_maskFrame.setTo( 0 );
		
		fillConvexPoly( m_maskFrame, m_newLeftEyePoly, 4, CV_RGB( 255, 255, 255 ) );
		fillConvexPoly( m_maskFrame, m_newRightEyePoly, 4, CV_RGB( 255, 255, 255 ) );
		fillConvexPoly( m_maskFrame, m_newNosePoly, 4, CV_RGB( 255, 255, 255 ) );
#endif

#if _DEBUG
		imwrite( "facemask.png", m_maskFrame ); // output mask frame
#endif

#if 1
		// increment frame count
		m_nFrameCount++;

		// if lot of frames has elapsed reset face detection
		if( m_nFrameCount >= FRAME_RESET_COUNTER )
		{
			// continue if frontal face is detected
			if ( -1 == detectFrontalFace ( m_currFrame, m_faceRect ) )
			{
				goto SKIP_RESET;
			}

#ifndef FACIAL_FEATURES
			shrinkRect( m_faceRect );
			convertRectToPoly( m_faceRect, m_facePoly );

			// create image mask for the face rectangle
			m_maskFrame.setTo( 0 );
			fillConvexPoly( m_maskFrame, m_facePoly, 4, CV_RGB( 255, 255, 255 ) );
#else
			// detect eyes within the face rectangle
			if( -1 == detectEyes( m_currFrame, m_leftEyeRect, m_rightEyeRect, m_noseRect ) )
			{
				goto SKIP_RESET;
			}

			// shrink rectangle
			shrinkRect( m_faceRect );
			shrinkRect( m_leftEyeRect );
			shrinkRect( m_rightEyeRect );
			shrinkRect( m_noseRect );

			// convert rectangles to polygons
			convertRectToPoly( m_faceRect, m_facePoly );
			convertRectToPoly( m_leftEyeRect, m_leftEyePoly );
			convertRectToPoly( m_rightEyeRect, m_rightEyePoly );
			convertRectToPoly( m_noseRect, m_nosePoly );

			// create mask for eyes and nose
			m_maskFrame.setTo( 0 );

			fillConvexPoly( m_maskFrame, m_leftEyePoly, 4, CV_RGB( 255, 255, 255 ) );
			fillConvexPoly( m_maskFrame, m_rightEyePoly, 4, CV_RGB( 255, 255, 255 ) );
			fillConvexPoly( m_maskFrame, m_nosePoly, 4, CV_RGB( 255, 255, 255 ) );

#endif

			// detect good features to track
			goodFeaturesToTrack( m_currFrame, m_featuresCurrent, 30, 0.01, 30, m_maskFrame); //calculate the features

			// reset transformation matrix
			m_affineMatrix = Mat::eye( 3, 3, CV_64FC1 );

			// reset frame counter
			m_nFrameCount = 0;
SKIP_RESET:
			m_nFrameCount;
		}
#endif
	} // end of else( fIsFirstFrame ) 

	return 0;
}

// perform 2D affine transformation on a given point
void CTrack2D::transformAffine( const cv::Point& point, cv::Point& newPoint )
{
	newPoint.x = int(m_affineMatrix.at<double>(0, 0) * (double)point.x
		+ m_affineMatrix.at<double>(0, 1) * (double)point.y
		+ m_affineMatrix.at<double>(0, 2));
	newPoint.y = int(m_affineMatrix.at<double>(1, 0) * (double)point.x
		+ m_affineMatrix.at<double>(1, 1) * (double)point.y
		+ m_affineMatrix.at<double>(1, 2));
}

// perform 2D affine transformation on a given polygon
void CTrack2D::transformAffine( const cv::Point poly[4], cv::Point newPoly[4] )
{
	for( int i = 0; i < 4; i++ )
	{
		transformAffine( poly[i], newPoly[i] );
	}
}

// shrink rectangle
void CTrack2D::shrinkRect( cv::Rect& rect )
{
	const float SHRINK_FACTOR = 0.15f;

	float deltaX = SHRINK_FACTOR * float(rect.width);
	float deltaY = SHRINK_FACTOR * float(rect.height);

	rect.x += int(0.5f * deltaX);
	rect.y += int(0.5f * deltaY);

	rect.width -= int(deltaX);
	rect.height -= int(deltaY);
}

// convert rectangle into a polygon of 4 points
void CTrack2D::convertRectToPoly( const cv::Rect& rect, cv::Point poly[4] )
{
	// for the time being consider clockwise. If doesn't work, do anti-clockwise
	poly[0].x = rect.x;
	poly[0].y = rect.y;

	poly[1].x = rect.x + rect.width;
	poly[1].y = rect.y;

	poly[2].x = rect.x + rect.width;
	poly[2].y = rect.y + rect.height;

	poly[3].x = rect.x;
	poly[3].y = rect.y + rect.height;
}

// make copy of polygon
void CTrack2D::copyPoly ( const cv::Point srcPoly[4], cv::Point dstPoly[4] )
{
	for( int i = 0; i < 4; i++ )
	{
		dstPoly[i] = srcPoly[i];
	}
}

// draw head pose info on input frame
int CTrack2D::drawPose( Mat& inputFrame )
{
	// draw face rectangle
	line( inputFrame, m_newFacePoly[0], m_newFacePoly[1], CV_RGB(255, 0, 0) );
	line( inputFrame, m_newFacePoly[1], m_newFacePoly[2], CV_RGB(255, 0, 0) );
	line( inputFrame, m_newFacePoly[2], m_newFacePoly[3], CV_RGB(255, 0, 0) );
	line( inputFrame, m_newFacePoly[3], m_newFacePoly[0], CV_RGB(255, 0, 0) );

#ifdef FACIAL_FEATURES
	// draw left eye
	line( inputFrame, m_newLeftEyePoly[0], m_newLeftEyePoly[1], CV_RGB(255, 255, 0) );
	line( inputFrame, m_newLeftEyePoly[1], m_newLeftEyePoly[2], CV_RGB(255, 255, 0) );
	line( inputFrame, m_newLeftEyePoly[2], m_newLeftEyePoly[3], CV_RGB(255, 255, 0) );
	line( inputFrame, m_newLeftEyePoly[3], m_newLeftEyePoly[0], CV_RGB(255, 255, 0) );

	// draw right eye
	line( inputFrame, m_newRightEyePoly[0], m_newRightEyePoly[1], CV_RGB(0, 255, 0) );
	line( inputFrame, m_newRightEyePoly[1], m_newRightEyePoly[2], CV_RGB(0, 255, 0) );
	line( inputFrame, m_newRightEyePoly[2], m_newRightEyePoly[3], CV_RGB(0, 255, 0) );
	line( inputFrame, m_newRightEyePoly[3], m_newRightEyePoly[0], CV_RGB(0, 255, 0) );

	// draw nose
	line( inputFrame, m_newNosePoly[0], m_newNosePoly[1], CV_RGB(0, 0, 255) );
	line( inputFrame, m_newNosePoly[1], m_newNosePoly[2], CV_RGB(0, 0, 255) );
	line( inputFrame, m_newNosePoly[2], m_newNosePoly[3], CV_RGB(0, 0, 255) );
	line( inputFrame, m_newNosePoly[3], m_newNosePoly[0], CV_RGB(0, 0, 255) );
#endif

	//Draw lines connecting previous position and current position
	for( size_t i = 0; i < m_featuresFound.size(); i++ )
	{
		if( m_featuresFound[i] )
		{
			line( inputFrame, 
				Point( (int)m_featuresPrevious[i].x - 3, (int)m_featuresPrevious[i].y - 3),
				Point( (int)m_featuresPrevious[i].x + 3, (int)m_featuresPrevious[i].y + 3),
				CV_RGB( 255, 0, 0 ) );

			line( inputFrame, 
				Point( (int)m_featuresPrevious[i].x - 3, (int)m_featuresPrevious[i].y + 3),
				Point( (int)m_featuresPrevious[i].x + 3, (int)m_featuresPrevious[i].y - 3),
				CV_RGB( 255, 0, 0 ) );
			//circle( inputFrame, m_featuresPrevious[i], 2, CV_RGB( 255, 0, 0 ) );
            line( inputFrame, m_featuresPrevious[i], m_featuresNextPos[i], CV_RGB( 0, 255, 0 ) );
		}
	}

	ostringstream strStream;
	strStream << "Vx = " << m_xvelocity << ", Vy = " << m_yvelocity << ", Omega = " << 57.272272 * m_omega;
	string strText = strStream.str();
	putText( inputFrame, strText, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 255, 0), 2 );

	return 0;
}

void CTrack2D::getVelocities (double &xvel, double &yvel, double &thetavel)
{
	xvel = m_xvelocity;
	yvel = m_yvelocity;
	thetavel = m_omega;
}
