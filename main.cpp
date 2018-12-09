/****************************************************************************\
* Vorlage fuer das Praktikum "Graphische Datenverarbeitung" WS 2018/19
* FB 03 der Hochschule Niedderrhein
* Regina Pohle-Froehlich
*
* Der Code basiert auf den c++-Beispielen der Bibliothek royale
\****************************************************************************/

/****************************************************************************\
* Modified to work with a webcam instead of the royale depth camera
* most code added to this sample should also work with the images provided
* by the depth camera (keep in mind that some functions, such as detecting FPS
* work differently with the royale)
* commented-out or in #if 0 #endif blocks wrapped lines are only required
* to use the depth camera and have been left in the code to identify
* the relative positions for copying code into the final assingment

* !!IMPORTANT!!
* grayImage and zImage have been converted to 32bit floating point array
* to match the input from the royale camera. Therefore, the image must be
* converted into an 8-bit array in order to apply historgram equalisation and
* to perform functions, such as imShow()!
* syntax source.convertTo(dest, type)
* MeowMix
\****************************************************************************/

// NOTES
// cv::threshold(src, dest, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

// #include <royale.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <exception>


#define ENTER 13
#define Evaluation 1
#define RECORDVIDEO 2
#define PLAYVIDEO 3


using namespace std;
using namespace cv;

class MyListener // : public royale::IDepthDataListener 
{

public:
	// L2A1 constructur to init frameCounter
	MyListener() {
		this->frameCounter = 0;
	}

#if 0
	void onNewData(const royale::DepthData *data)
	{
		// this callback function will be called for every new depth frame

		std::lock_guard<std::mutex> lock(flagMutex);
		zImage.create(cv::Size(data->width, data->height), CV_32FC1);
		grayImage.create(cv::Size(data->width, data->height), CV_32FC1);
		zImage = 0;
		grayImage = 0;
		int k = 0;
		for (int y = 0; y < zImage.rows; y++)
		{
			for (int x = 0; x < zImage.cols; x++)
			{
				auto curPoint = data->points.at(k);
				if (curPoint.depthConfidence > 0)
				{
					// if the point is valid
					zImage.at<float>(y, x) = curPoint.z;
					grayImage.at<float>(y, x) = curPoint.grayValue;
				}
				k++;
			}
		}

		cv::Mat temp = zImage.clone();
		undistort(temp, zImage, cameraMatrix, distortionCoefficients);
		temp = grayImage.clone();
		undistort(temp, grayImage, cameraMatrix, distortionCoefficients);
	}

	void setLensParameters(const royale::LensParameters &lensParameters)
	{
		// Construct the camera matrix
		// (fx   0    cx)
		// (0    fy   cy)
		// (0    0    1 )
		cameraMatrix = (cv::Mat1d(3, 3) << lensParameters.focalLength.first, 0, lensParameters.principalPoint.first,
			0, lensParameters.focalLength.second, lensParameters.principalPoint.second,
			0, 0, 1);

		// Construct the distortion coefficients
		// k1 k2 p1 p2 k3
		distortionCoefficients = (cv::Mat1d(1, 5) << lensParameters.distortionRadial[0],
			lensParameters.distortionRadial[1],
			lensParameters.distortionTangential.first,
			lensParameters.distortionTangential.second,
			lensParameters.distortionRadial[2]);
	}
#endif
	// DO NOT CHANGE THE CODE BELOW
	void startCapture(bool *captureFlag) {
		WebCam = std::thread(&MyListener::onNewData, this, std::ref(*captureFlag));
	}

	void stopCapture(bool *captureFlag) {
		*captureFlag = false;
		if (WebCam.joinable()) {
			WebCam.join();
		}
	}

	void onNewData(bool &captureFlag) {
		// this callback function will be called for every new depth frame
		std::lock_guard<std::mutex> lock(flagMutex);
		cv::VideoCapture camVideo(0);
		int width = camVideo.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = camVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
		double FPS = camVideo.get(CV_CAP_PROP_FPS);

		zImage.create(cv::Size(width, height), CV_32FC1);
		grayImage.create(cv::Size(width, height), CV_32FC1);
		zImage = 0;
		grayImage = 0;

		if (!camVideo.isOpened()) {
			std::cerr << "Error opening webcam. Make sure it's connected and disable your antivirus" << std::endl;
			cv::destroyAllWindows();
		}
		else {
			while (captureFlag) {
				cv::Mat tmp(cv::Size(width, height), CV_8U);
				camVideo >> tmp;
				cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
				tmp.convertTo(zImage, CV_32FC1);

				grayImage = zImage.clone();
				// FROM HERE ON CHANGES CAN BE APPLIED AGAIN

				// my code
				showImage();

				// assignment 5
				if (mode == RECORDVIDEO) {
					if (zVideo.isOpened())
						zVideo << zImage;		// assingment 5: read zImage into zVideo
					if (grayVideo.isOpened())
						grayVideo << grayImage;	// assingment 5: read grayImage into grayVideo
				}
			}
		}
	}

	// L2A1
	void createAverageImage() {
		if (frameCounter == 0) {
			accGrayImage = Mat::zeros(grayImage.size(), CV_64F);
		}

		frameCounter = (++frameCounter) % 20;	// ringbuffer, ranging from 0-19
		// accumulateWeighted(grayImage, accGrayImage, 0.5);
		accumulate(grayImage, accGrayImage);

		if (frameCounter == 0) {
			accGrayImage.convertTo(accFrameGrayImage, CV_8U, 1.0 / 20);

			imshow("20 Frames", accFrameGrayImage);
		}
	}

	// manual hisogram equalisation (assignment 1)
// old ugly solution
#if 0
	void spreadHistogram(Mat *pic) {
		Mat zeroFreeMask = Mat::zeros(pic->size(), pic->type());	// initialise empty mask
		double min = 0.0;
		double max = 0.0;

		if (!pic->empty()) {
			// compare depth image against zeroscale to remove zero values (CMP_GT)
			compare(*pic, Scalar(0, 0, 0, 0), zeroFreeMask, CMP_GT);

			minMaxLoc(*pic, &min, &max, NULL, NULL, zeroFreeMask);

			// lineare scaling using min and max calculated via minMaxLoc
			for (int i = 0; i < pic->rows; i++) {
				for (int j = 0; j < pic->cols; j++) {
					// exclude pixel value 0
					if (pic->at<float>(i, j) > 0) {
						pic->at<float>(i, j) = (pic->at<float>(i, j) - min) * (255 / (max - min)); // formular from the lecture XD
					}
				}
			}
		}

		else {
			perror("Hisogramequalisation failed!");
		}
	}
#endif

	void spreadHistogram(Mat *pic) {
		Mat nonZeroMask = pic->clone();
		compare(*pic, 0, nonZeroMask, CV_CMP_NE);

		double min, max, scale, shift;
		minMaxLoc(*pic, &min, &max, NULL, NULL, nonZeroMask);
		scale = 255.0 / (max - min);
		shift = -255.0 * min / (max - min);
		convertScaleAbs(*pic, *pic, scale, shift);
	}

	// assignment 1
	void showImage() {
		if (!zImage.empty()) {
			spreadHistogram(&zImage);
			zImage.convertTo(zImage, CV_8U);
			applyColorMap(zImage, zImage, COLORMAP_RAINBOW);

			imshow("zMeow", zImage);

			//			if (frameCounter == 0) {
			//				imshow("zMeow", zImage);
			//			}
		}
		else {
			perror("Displaying zImage failed!");
		}

		if (!grayImage.empty()) {
			spreadHistogram(&grayImage);
			grayImage.convertTo(grayImage, CV_8U);

			// L2A1
			createAverageImage();

			imshow("grayMeow", grayImage);

			// filter tests
			blur(grayImage, avgGrayImage, Size(5, 5));
			imshow("Average", avgGrayImage);

			medianBlur(grayImage, medianGrayImage, 5);
			imshow("Median", medianGrayImage);

			// why not bilateral
			bilateralFilter(grayImage, bilateralGrayImage, 9, 75, 75, BORDER_CONSTANT);
			imshow("Bilaterial", bilateralGrayImage);

			//			if (frameCounter == 0) {
			//				imshow("grayMeow", grayImage);
			//			}
		}
		else {
			perror("Displaying grayImage failed!");
		}
		waitKey(1);
	}

	// assingment 5
	void openVideoWriter(Size size, string zName, string grayName, double fps) {
		zVideo.open(zName, CV_FOURCC('M', 'J', 'P', 'G'), fps, size, true);
		grayVideo.open(grayName, CV_FOURCC('M', 'J', 'P', 'G'), fps, size, false);
	}

	void closeVideoWriter() {
		if (zVideo.isOpened()) {
			zVideo.release();
		}
		if (grayVideo.isOpened()) {
			grayVideo.release();
		}

	}

	// assingment 3 / 4 /5
	void videoHandler(string prefix, Size size, uint16_t framerate, bool streamCapture) {
		cout << "Maximum Imagesize is: " << size << endl;

		string zName = prefix + "_depth.avi";
		string grayName = prefix + "_gray.avi";

		openVideoWriter(size, zName, grayName, framerate);
	}

	// assignment 6
	void openStreamCapture(string *prefix) {
		zStream.open(*prefix + "_depth.avi");
		grayStream.open(*prefix + "_gray.avi");
	}

	void closeStreamCapture() {
		if (zStream.isOpened()) {
			zStream.release();
		}
		if (grayStream.isOpened()) {
			grayStream.release();
		}
	}

	void showCapture(string prefix, double framerate) {
		openStreamCapture(&prefix);
		// assignment 6: read, grab frames, display
		if (zStream.isOpened() && grayStream.isOpened()) {
			namedWindow("deepStream", cv::WINDOW_AUTOSIZE);
			namedWindow("grayStream", cv::WINDOW_AUTOSIZE);

			Mat z;
			Mat gray;
			while (zStream.grab() && grayStream.grab()) {
				zStream.retrieve(z);
				imshow("deepStream", z);

				grayStream.retrieve(gray);
				imshow("grayStream", gray);
				waitKey(framerate * 40);
			}
		}
		else {
			perror("Error Show Capture");
		}
		closeStreamCapture();
	}

	void setMode(int mode) {
		this->mode = mode;
	}

private:

	cv::Mat zImage, grayImage;
	cv::Mat cameraMatrix, distortionCoefficients;
	std::mutex flagMutex;

	std::thread WebCam;

	// my code
	VideoWriter zVideo;
	VideoWriter grayVideo;

	// assignment 6
	VideoCapture zStream;
	VideoCapture grayStream;

	// parameter for program
	// 1: Evaluation
	// 2: Record Video
	// 3: Play Video
	int mode;

	// L2A1
	int frameCounter;	// number of frames used to accumulate image, when 0 display image
	cv::Mat accGrayImage, accFrameGrayImage, avgGrayImage, medianGrayImage, bilateralGrayImage;
};

int main(int argc, char *argv[]) {
#if 0
	MyListener listener;

	// this represents the main camera device object
	std::unique_ptr<royale::ICameraDevice> cameraDevice;

	// the camera manager will query for a connected camera
	{
		royale::CameraManager manager;

		// try to open the first connected camera
		royale::Vector<royale::String> camlist(manager.getConnectedCameraList());
		std::cout << "Detected " << camlist.size() << " camera(s)." << std::endl;

		if (!camlist.empty())
		{
			cameraDevice = manager.createCamera(camlist[0]);
		}
		else
		{
			std::cerr << "No suitable camera device detected." << std::endl
				<< "Please make sure that a supported camera is plugged in, all drivers are "
				<< "installed, and you have proper USB permission" << std::endl;
			return 1;
		}

		camlist.clear();

	}
	// the camera device is now available and CameraManager can be deallocated here

	if (cameraDevice == nullptr)
	{
		// no cameraDevice available
		if (argc > 1)
		{
			std::cerr << "Could not open " << argv[1] << std::endl;
			return 1;
		}
		else
		{
			std::cerr << "Cannot create the camera device" << std::endl;
			return 1;
		}
	}

	// call the initialize method before working with the camera device
	auto status = cameraDevice->initialize();
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Cannot initialize the camera device, error string : " << getErrorString(status) << std::endl;
		return 1;
	}

	// retrieve the lens parameters from Royale
	royale::LensParameters lensParameters;
	status = cameraDevice->getLensParameters(lensParameters);
	if (status != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Can't read out the lens parameters" << std::endl;
		return 1;
	}

	listener.setLensParameters(lensParameters);

	// register a data listener
	if (cameraDevice->registerDataListener(&listener) != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error registering data listener" << std::endl;
		return 1;
	}

	// start capture mode
	if (cameraDevice->startCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error starting the capturing" << std::endl;
		return 1;
	}

	// stop capture mode
	if (cameraDevice->stopCapture() != royale::CameraStatus::SUCCESS)
	{
		std::cerr << "Error stopping the capturing" << std::endl;
		return 1;
	}
#endif
	MyListener listener;

	string prefix;

	bool captureFlag = true;	// true while WebCam is allowed to capture

	cv::namedWindow("zMeow", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("grayMeow", CV_WINDOW_AUTOSIZE);

	// start capture mode
	// equivalent to start capture from the original
	listener.startCapture(&captureFlag);

#if 0 // royalle
	uint16_t width, height, framerate;
	cameraDevice->getMaxSensorWidth(width);
	cameraDevice->getMaxSensorHeight(height);
	cameraDevice->getMaxFrameRate(framerate);

	Size size = Size(width, height);
#endif
	uint16_t width, height, framerate;
	width = 640;
	height = 480;
	framerate = 10;

	Size size = Size(width, height);

	// assingment 3 /4 / 6
	// input via modified properties through argc and argv
	if (argc > 1) {
		// assignment 3
		switch (stoi(argv[1])) {
		case Evaluation: // assignemtn 3: if 1 then print message "data evaluation"
			cout << "Call of evaluation method" << endl;
			listener.setMode(Evaluation);
			break;
		case RECORDVIDEO: // assignment 3 / 4: if 2 then record video, read prefix of video file name or take as parameter
			if (argc == 3) {
				prefix = argv[2];
			}
			else {
				cout << "Please enter a name for the video file" << endl;
				cin >> prefix;
			}
			listener.setMode(RECORDVIDEO);
			listener.videoHandler(prefix, size, framerate, false);
			break;
		case PLAYVIDEO:	// assignment 6: same as assignment 4 and 5 but dispaly video as well 
			if (argc == 3) {
				prefix = argv[2];
			}
			else {
				cout << "Please enter a name for the video file" << endl;
				cin >> prefix;
			}
			listener.showCapture(prefix, framerate);
			break;
		default:
			cout << "DEBUG: No additional parameters passed" << endl;
		}
	}

	while (waitKey(0) != ENTER) {
		// nothing happens here because we are waiting for enter to be pressed
	}


	// stop capture mode
	// equivalent to start capture from the original
	listener.stopCapture(&captureFlag);

	return 0;
}
