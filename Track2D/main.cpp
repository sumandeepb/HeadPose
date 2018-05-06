#include <opencv2/opencv.hpp>
#include "Track2D.h"
#include "CVPlot.h"

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

	// tracker object
	CTrack2D		cTracker;

	// plot parameters
	const int PLOTRANGE = 256;
	const double DC_OFFSET = 128.0f;
	const double ROT_SCALE = 10.0f * 57.272272f;
	const double TRANS_SCALE = 10.0f;
	int	nPlotPos = 0;
	unsigned char	omega[PLOTRANGE];
	unsigned char	velX[PLOTRANGE];
	unsigned char	velY[PLOTRANGE];
	memset( omega, 0, sizeof(unsigned char) * PLOTRANGE );
	memset( velX, 0, sizeof(unsigned char) * PLOTRANGE );
	memset( velY, 0, sizeof(unsigned char) * PLOTRANGE );

	// run untill 'ESCAPE' key is pressed
#if _DEBUG
	while ( 27 != cvWaitKey( 0 ) )
#else
	while ( 27 != cvWaitKey( 1 ) )
#endif
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
			
			if( -1 == cTracker.trackNewFrame( cInputFrame2 ) )
			{
				cerr << "Error: Tracking lost" << endl;
			}

			// draw pose axes onto the input frame
			if ( -1 == cTracker.drawPose ( cInputFrame2 ) )
			{
				cerr << "ERROR: Failed to draw Head Pose axes" << endl;
				continue;
			}

			// display
			imshow( "Head Pose", cInputFrame2 );

			// plot
			double xvel, yvel, thetavel;
			cTracker.getVelocities (xvel, yvel, thetavel);
			omega[nPlotPos] = ROT_SCALE * thetavel + DC_OFFSET;
			velX[nPlotPos] = TRANS_SCALE * xvel + DC_OFFSET;
			velY[nPlotPos] = TRANS_SCALE * yvel + DC_OFFSET;
			
			CvPlot::clear("Omega");
			CvPlot::plot("Omega", omega, PLOTRANGE, 1, 0, 255, 0);
			CvPlot::label("Omega");

			CvPlot::clear("Vx");
			CvPlot::plot("Vx", velX, PLOTRANGE, 1, 255, 0, 0);
			CvPlot::label("Vx");

			CvPlot::clear("Vy");
			CvPlot::plot("Vy", velY, PLOTRANGE, 1, 0, 0, 255);
			CvPlot::label("Vy");

			nPlotPos++;
			if( nPlotPos >= PLOTRANGE)
			{
				nPlotPos = 0;
			}
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