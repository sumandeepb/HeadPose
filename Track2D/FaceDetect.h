#pragma once

#include <opencv2/opencv.hpp>

int detectFrontalFace( const cv::Mat& imageFrame, cv::Rect& faceRect );
int detectEyes( const cv::Mat& imageFrame, cv::Rect& leftEyeRect, cv::Rect& rightEyeRect, cv::Rect& noseRect );