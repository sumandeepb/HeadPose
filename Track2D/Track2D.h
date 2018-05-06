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

#pragma once

#include "opencv2\opencv.hpp"
//#include "Vector2D.h"

//#define FACIAL_FEATURES 			// whether to use facial features instead of whole face
#define FRAME_RESET_COUNTER	100		// reset the face detector after the frame count

class CTrack2D
{
protected:
	cv::Mat		m_prevFrame;			// previous frame
	cv::Mat		m_currFrame;			// current frame
	cv::Mat		m_maskFrame;			// face detector mask
	bool		m_fIsFirstFrame;		// flag to detet the very first frame before tracking begins
	int			m_nFrameCount;			// number of frames elapsed sinse face detection

	cv::Rect	m_faceRect;				// face rectangle
	cv::Rect	m_leftEyeRect;			// left eye rectangle
	cv::Rect	m_rightEyeRect;			// right eye rectangle
	cv::Rect	m_noseRect;				// nose rectangle

	cv::Point	m_facePoly[4];			// face polygon
	cv::Point	m_leftEyePoly[4];		// left eye polygon
	cv::Point	m_rightEyePoly[4];		// right eye polygon
	cv::Point	m_nosePoly[4];			// nose polygon

	std::vector<cv::Point2f>	m_featuresPrevious;	// keypoints in previous frame
    std::vector<cv::Point2f>	m_featuresCurrent;	// keypoints in current frame
    std::vector<cv::Point2f>	m_featuresNextPos;	// predicted keypoints in next frame
	std::vector<unsigned char>	m_featuresFound;	// if features could be found and predicted
	cv::Mat						m_flowErr;			// error in block matching of motion vector

	cv::Mat		m_affineMatrix;			// affine transformation matrix for face rectangle
	double		m_omega;				// face rectangle angular velocity
	//double		m_zoom;	
	double		m_xvelocity;			// face rectangle horizontal velocity
	double		m_yvelocity;			// face rectangle vertical velocity
	//double		m_xdisplacement;
	//double		m_ydisplacement;
	//double		m_scale;

	cv::Point	m_newFacePoly[4];		// face rectangle
	cv::Point	m_newLeftEyePoly[4];	// left eye rectangle
	cv::Point	m_newRightEyePoly[4];	// right eye rectangle
	cv::Point	m_newNosePoly[4];		// nose rectangle

protected:
	void transformAffine( const cv::Point& point, cv::Point& newPoint );	// perform 2D affine transformation on a given point
	void transformAffine( const cv::Point poly[4], cv::Point newPoly[4] );	// perform 2D affine transformation on a given polygon
	void shrinkRect( cv::Rect& rect );										// shrink rectangle
	void convertRectToPoly( const cv::Rect& rect, cv::Point poly[4] ); // convert rectangle into a polygon of 4 points
	void copyPoly ( const cv::Point srcPoly[4], cv::Point dstPoly[4] ); // make copy of polygon

public:
	CTrack2D();
	int trackNewFrame( cv::Mat	&inputFrame );		// track face in continuous video frames
	int drawPose( cv::Mat& inputFrame );			// draw head pose info on input frame
	void getVelocities (double &xvel, double &yvel, double &thetavel); // get optical flow velocities
};
