#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

using namespace std;

// OpenCV Camera Calibration Test with chessboard.
// https://docs.opencv.org/master/d4/d94/tutorial_camera_calibration.html
// https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp

#define IMAGE_NUM  (13)						// How many frames to use, for calibration.
#define PAT_ROW    (6)						// Number of inner corners per a item row.
#define PAT_COL    (9)						// Number of inner corners per a item column.
#define CHESS_SIZE (50.0)					// The size of a square in some user defined metric system (pixel, millimeter)

#define USE_NEW_CALIBRATION_METHOD (true)	// If your calibration board is inaccurate, unmeasured, roughly planar targets
											// (Checkerboard patterns on paper using off-the-shelf printers are the most convenient calibration targets and most of them are not accurate enough.),
											// a method from [219] can be utilized to dramatically improve the accuracies of the estimated camera intrinsic parameters. 
											// https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga11eeb16e5a458e1ed382fb27f585b753
#define GRID_WIDTH (400.0)					// the measured distance between top-left (0, 0, 0) and top-right (s.squareSize*(s.boardSize.width-1), 0, 0) corners of the pattern grid points.

#define USE_FIND_CHESSBOARD_CORNERS_SB_METHOD (true)	// Determines if use findChessboardCornersSB method. (More accurate than the findChessboardCorners and cornerSubPix methods)
														// https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91
#define ENABLE_CORNER_SUB_PIX (true)		// Determines if enable CornerSubPix method. (Improve the found corners' coordinate accuracy for chessboard)


int main(int argc, char *argv[])
{
	int i, j, k;
	vector<cv::Mat> srcImages;

	// Load calibration images.
	for (i = 0; i < IMAGE_NUM; i++)
	{
		ostringstream ostr;
		ostr << "../calibration_images_right00-12\\right" << setfill('0') << setw(2) << i << ".jpg";
		cv::Mat src = cv::imread(ostr.str());
		if (src.empty())
		{
			cerr << "cannot load image file : " << ostr.str() << endl;
		}
		else
		{
			srcImages.push_back(src);
		}
	}


	// Calc board corner positions.
	vector<cv::Point3f> object;
	for (j = 0; j < PAT_ROW; j++)
	{
		for (k = 0; k < PAT_COL; k++)
		{
			object.push_back(cv::Point3f(k * CHESS_SIZE, j * CHESS_SIZE, 0));
		}
	}

	float grid_width = CHESS_SIZE * (PAT_COL - 1);
	bool release_object = false;
	if (USE_NEW_CALIBRATION_METHOD) {
		grid_width = GRID_WIDTH;
		release_object = true;
	}
	object[PAT_COL - 1].x = object[0].x + grid_width;
	vector<cv::Point3f> newObjPoints = object;

	vector<vector<cv::Point3f>> obj_points;
	for (i = 0; i < IMAGE_NUM; i++)
	{
		obj_points.push_back(object);
	}


	// Find chessboard corners.
	int findCornersFlags = USE_FIND_CHESSBOARD_CORNERS_SB_METHOD ? cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY : cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
	int winSize = 11;
	cv::Size pattern_size = cv::Size2i(PAT_COL, PAT_ROW);
	vector<cv::Point2f> corners;
	vector<vector<cv::Point2f>> img_points;
	int found_num = 0;
	cv::namedWindow("Calibration", cv::WINDOW_AUTOSIZE);
	for (i = 0; i < IMAGE_NUM; i++)
	{
		auto found = false;
		if (USE_FIND_CHESSBOARD_CORNERS_SB_METHOD)
		{
			found = cv::findChessboardCornersSB(srcImages[i], pattern_size, corners, findCornersFlags);
		}
		else
		{
			found = cv::findChessboardCorners(srcImages[i], pattern_size, corners, findCornersFlags);
		}

		if (found)
		{
			cout << setfill('0') << setw(2) << i << "... ok" << endl;
			found_num++;
		}
		else
		{
			cerr << setfill('0') << setw(2) << i << "... fail" << endl;
		}

		if (!USE_FIND_CHESSBOARD_CORNERS_SB_METHOD && ENABLE_CORNER_SUB_PIX)
		{
			cv::Mat src_gray = cv::Mat(srcImages[i].size(), CV_8UC1);
			cv::cvtColor(srcImages[i], src_gray, cv::COLOR_BGR2GRAY);
			cv::cornerSubPix(src_gray, corners, cv::Size(winSize, winSize), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.0001));
		}

		cv::drawChessboardCorners(srcImages[i], pattern_size, corners, found);
		img_points.push_back(corners);

		cv::imshow("Calibration", srcImages[i]);
		cv::waitKey(0);
	}
	cv::destroyWindow("Calibration");

	if (found_num != IMAGE_NUM)
	{
		cerr << "Calibration Images are insufficient." << endl;
		return -1;
	}


	// Find intrinsic and extrinsic camera parameters.
	double aspect_raito = 1.0;
	int calibrationFlags = cv::CALIB_FIX_PRINCIPAL_POINT | cv::CALIB_ZERO_TANGENT_DIST | cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5;
	cv::Mat cam_mat = cv::Mat::eye(3, 3, CV_64F);
	if (calibrationFlags & cv::CALIB_FIX_ASPECT_RATIO)
		cam_mat.at<double>(0, 0) = aspect_raito;
	cv::Mat dist_coefs = cv::Mat::zeros(8, 1, CV_64F);
	vector<cv::Mat> rvecs, tvecs;

	int iFixedPoint = -1;
	if (release_object)
		iFixedPoint = PAT_COL - 1;
	double repErr = cv::calibrateCameraRO(
		obj_points,
		img_points,
		srcImages[0].size(),
		iFixedPoint,
		cam_mat,
		dist_coefs,
		rvecs,
		tvecs,
		newObjPoints,
		calibrationFlags | cv::CALIB_USE_LU
	);

	if (release_object) {
		cout << "New board corners: " << endl;
		cout << newObjPoints[0] << endl;
		cout << newObjPoints[PAT_COL - 1] << endl;
		cout << newObjPoints[PAT_COL * (PAT_ROW - 1)] << endl;
		cout << newObjPoints.back() << endl;
	}


	// Print camera parameters to the output file.
	cv::FileStorage fs("../out_camera_parameters.xml", cv::FileStorage::WRITE);
	if (!fs.isOpened())
	{
		cerr << "File can not be opened." << endl;
		return -1;
	}

	fs << "intrinsic" << cam_mat;
	fs << "distortion" << dist_coefs;
	fs << "repErr" << repErr;
	fs << "USE_FIND_CHESSBOARD_CORNERS_SB_METHOD" << USE_FIND_CHESSBOARD_CORNERS_SB_METHOD;
	fs << "findCornersFlags" << findCornersFlags;
	fs << "ENABLE_CORNER_SUB_PIX" << ENABLE_CORNER_SUB_PIX;
	fs << "USE_NEW_CALIBRATION_METHOD" << USE_NEW_CALIBRATION_METHOD;
	fs << "calibrationFlags" << calibrationFlags;

	fs.release();



	//
	//cout << "obj_points: " << obj_points.size() << endl;
	//cout << "obj_points: " << obj_points[0] << endl;
	//cout << "img_points: " << img_points.size() << endl;
	//cout << "img_points: " << img_points[0] << endl;
	//cout << "srcImages[0].size(): " << srcImages[0].size() << endl;
	cout << "intrinsic: " << cam_mat << endl;
	cout << "distortion: " << dist_coefs << endl;
	cout << "repErr: " << repErr << endl;
	cout << "USE_FIND_CHESSBOARD_CORNERS_SB_METHOD: " << USE_FIND_CHESSBOARD_CORNERS_SB_METHOD << endl;
	cout << "findCornersFlags: " << findCornersFlags << endl;
	cout << "ENABLE_CORNER_SUB_PIX: " << ENABLE_CORNER_SUB_PIX << endl;
	cout << "USE_NEW_CALIBRATION_METHOD: " << USE_NEW_CALIBRATION_METHOD << endl;
	cout << "calibrationFlags: " << calibrationFlags << endl;
	//

	return 0;
}