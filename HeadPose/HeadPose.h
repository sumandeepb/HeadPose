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

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>

// class to estimate head pose from a sequence of image frames
class CHeadPoseEstimator {
protected:	// data
	cv::Mat		m_cameraModel;				// camera intrinsic matrix (A)
	cv::Mat		m_cameraRVec;				// camera pose rotation vector
	cv::Mat		m_cameraRot;				// camera pose rotation matrix (R)
	cv::Mat		m_cameraTran;				// camera pose translation (T)
	cv::Mat		m_cameraPers;				// perspective projection matrix

	bool		m_fKeyFrameDetected;		// status of key frame detection
	cv::Rect	m_faceRect;					// face rectangle in key frame
	cv::Rect	m_leftEyeRect;				// left eye rectangle
	cv::Rect	m_rightEyeRect;				// right eye rectangle

	cv::Mat		m_keyFrame;					// key frame (grayscale)
	std::vector<cv::KeyPoint> m_keyFrameKeyPoints;	// keypoints (m)
	cv::Mat		m_keyFrameDescriptors;		// keypoint descriptors
	std::vector<cv::Point2f> m_keyFrame2DPoints;	// keypoint 2D coordinates (m)
	std::vector<cv::Point3f> m_keyFrame3DPoints;	// keypoints on 3D Model (M)

	cv::Mat		m_lastFrame;				// last frame (grayscale) frame index = t - 1
	std::vector<cv::KeyPoint> m_lastFrameKeyPoints;	// keypoints (m)
	cv::Mat		m_lastFrameDescriptors;		// keypoint descriptors
	std::vector<cv::Point2f> m_lastFrame2DPoints;	// keypoint 2D coordinates (m)
	std::vector<cv::Point3f> m_lastFrame3DPoints;	// keypoints on 3D Model (M)

	cv::Mat		m_currFrame;				// current frame (grayscale) frame index = t
	//cv::Mat		m_HeadPose;					// current head pose relative to the keyframe pose

	cv::Mat			m_maskFrame;			// detector mask
	cv::BRISK		m_BRISKDetector;		// Binary Robust Invariant Scalable Keypoint Detector
	cv::BFMatcher	m_BFMatcher;			// Brute Force Matcher

public:		// exposed operations
	CHeadPoseEstimator();
	~CHeadPoseEstimator();

	int addNewFrame( const cv::Mat& inputFrame );	// frame feed from video capture device
	int getHeadPose();								// get head pose estimation
	int drawPose( cv::Mat& inputFrame );			// draw head pose coordinate axes on input frame

protected:	// internal support functions
	cv::Point3f model2Dto3D( const cv::Point2f& m, int method = 0 );		// keyframe points to 3D Model
	cv::Point3f project2Dto3D( const cv::Point2f& m );		// backproject 2D keypoints to 3D Model
};

int Write3DPointsToObj( const char strFileName[], std::vector<cv::Point3f>& points3D ); // write 3D points to Obj File 
