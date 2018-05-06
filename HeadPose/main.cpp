#include <opencv2/opencv.hpp>
#include "HeadPose.h"

using namespace std;
using namespace cv;

// main function
int main( int argc, char* argv[] )
{
	// constant defines
	const int IDEALWIDTH = 640;

	// get camera ID from command line argument
	int nCamID = atoi( argv[1] );

	// Video Capture object
	VideoCapture cCapDevice;
	Mat			 cInputFrame, cInputFrame2;

	if ( nCamID < 0 ) // if invalid camera ID, use video file as input  
	{
		cCapDevice.open( argv[2] );
	}
	else // for valid camera ID, use live camera feed as input
	{
		cCapDevice.open( nCamID );
	}

	// error in opening capture device (camera/video)
	if ( !cCapDevice.isOpened() )
	{
		cerr << "ERROR: Failed to open Video Capture device" << endl;
		return -1;
	}

	// Head pose estimator object
	CHeadPoseEstimator	cHeadPose;

	// run untill 'ESCAPE' key is pressed
	while ( 27 != cvWaitKey( 1 ) )
	{
		if( cCapDevice.read( cInputFrame ) )
		{
			double fx, fy;
			double dAspectRatio;

			dAspectRatio = (double)cInputFrame.cols / (double)cInputFrame.rows;
			fx = (double)IDEALWIDTH / (double)cInputFrame.cols;
			fy = fx;

			if( fx >= 1.0 )
			{
				// upsizing image for uniform performance
				resize( cInputFrame, cInputFrame2, Size(), fx, fy, INTER_LINEAR );
			}
			else
			{
				// downsizing of input frame for faster processing
				resize( cInputFrame, cInputFrame2, Size(), fx, fy, INTER_AREA );
			}
			
			// send captured frame to head pose estimator
			if ( -1 == cHeadPose.addNewFrame ( cInputFrame2 ) )
			{
				cerr << "ERROR: Failed to add frame to Head Pose Estimator" << endl;
				continue;
			}

			// draw pose axes onto the input frame
			if ( -1 == cHeadPose.drawPose ( cInputFrame2 ) )
			{
				cerr << "ERROR: Failed to draw Head Pose axes" << endl;
				continue;
			}

			// display
			imshow( "Head Pose", cInputFrame2 );
		}
		else
		{
			cerr << "ERROR: End of video stream" << endl;
			break;
		}
	}

	destroyAllWindows();
	return 0;
}