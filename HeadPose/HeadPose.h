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
