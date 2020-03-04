using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// OpenCV Camera Calibration Test with chessboard.
// https://docs.opencv.org/master/d4/d94/tutorial_camera_calibration.html
// https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/camera_calibration/camera_calibration.cpp
public class OpenCVCameraCalibrationTest : MonoBehaviour
{

    const int IMAGE_NUM = 13;                       // How many frames to use, for calibration.
    const int PAT_ROW = 6;                          // Number of inner corners per a item row.
    const int PAT_COL = 9;                          // Number of inner corners per a item column.
    const float CHESS_SIZE = 50f;                   // The size of a square in some user defined metric system (pixel, millimeter)
    const bool USE_NEW_CALIBRATION_METHOD = true;   // If your calibration board is inaccurate, unmeasured, roughly planar targets
                                                    // (Checkerboard patterns on paper using off-the-shelf printers are the most convenient calibration targets and most of them are not accurate enough.),
                                                    // a method from [219] can be utilized to dramatically improve the accuracies of the estimated camera intrinsic parameters.
                                                    // https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#ga11eeb16e5a458e1ed382fb27f585b753
    const float GRID_WIDTH = 400f;                  // the measured distance between top-left (0, 0, 0) and top-right (s.squareSize*(s.boardSize.width-1), 0, 0) corners of the pattern grid points.

    const bool USE_FIND_CHESSBOARD_CORNERS_SB_METHOD = true; // Determines if use findChessboardCornersSB method. (More accurate than the findChessboardCorners and cornerSubPix methods)
                                                             // https://docs.opencv.org/4.2.0/d9/d0c/group__calib3d.html#gad0e88e13cd3d410870a99927510d7f91
    const bool ENABLE_CORNER_SUB_PIX = true;        // Determines if enable CornerSubPix method. (Improve the found corners' coordinate accuracy for chessboard)

    const int findChessboardCornersFlags =
        Calib3d.CALIB_CB_ADAPTIVE_THRESH |
        Calib3d.CALIB_CB_NORMALIZE_IMAGE |
        Calib3d.CALIB_CB_FAST_CHECK |
        0;

    const int findChessboardCornersSBFlags =
        Calib3d.CALIB_CB_NORMALIZE_IMAGE | 
        Calib3d.CALIB_CB_EXHAUSTIVE | 
        Calib3d.CALIB_CB_ACCURACY |
        0;

    const int calibrateCameraROFlags =
        Calib3d.CALIB_FIX_PRINCIPAL_POINT |
        Calib3d.CALIB_FIX_ASPECT_RATIO |
        Calib3d.CALIB_ZERO_TANGENT_DIST |
        Calib3d.CALIB_FIX_K4 |
        Calib3d.CALIB_FIX_K5 |
        0;

    // Use this for initialization
    IEnumerator Start()
    {

        Texture2D texture = new Texture2D(640, 480, TextureFormat.RGBA32, false);
        gameObject.GetComponent<Renderer>().material.mainTexture = texture;
        gameObject.transform.localScale = new Vector3(texture.width, texture.height, 1);


        int i, j, k;
        List<Mat> srcImages = new List<Mat>();

        // Load calibration images.
        for (i = 0; i < IMAGE_NUM; i++)
        {
            string path = Utils.getFilePath("calibration_images_right00-12/right" + String.Format("{0:00}", i) + ".jpg");
            Mat src = Imgcodecs.imread(path);
            if (src.empty())
            {
                Debug.Log("cannot load image file : " + path);
            }
            else
            {
                srcImages.Add(src);
            }
        }


        // Calc board corner positions.
        MatOfPoint3f objectPt = new MatOfPoint3f(new Mat(PAT_ROW * PAT_COL, 1, CvType.CV_32FC3));
        for (j = 0; j < PAT_ROW; j++)
        {
            for (k = 0; k < PAT_COL; k++)
            {
                objectPt.put(PAT_COL * j + k, 0, new float[] { k * CHESS_SIZE, j * CHESS_SIZE, 0f });
            }
        }

        float grid_width = CHESS_SIZE * (PAT_COL - 1);
        bool release_object = false;
        if (USE_NEW_CALIBRATION_METHOD)
        {
            grid_width = GRID_WIDTH;
            release_object = true;
        }
        float[] tlPt = new float[3]; // top-left point
        objectPt.get(0, 0, tlPt);
        float[] trPt = new float[3]; // top-right point
        objectPt.get(PAT_COL - 1, 0, trPt);
        trPt[0] = tlPt[0] + grid_width;
        objectPt.put(PAT_COL - 1, 0, trPt);
        Mat newObjPoints = objectPt.clone();

        List<Mat> obj_points = new List<Mat>();
        for (i = 0; i < IMAGE_NUM; i++)
        {
            obj_points.Add(objectPt.clone());
        }


        // Find chessboard corners.
        int findCornersFlags = USE_FIND_CHESSBOARD_CORNERS_SB_METHOD ? findChessboardCornersSBFlags : findChessboardCornersFlags;
        int winSize = 11;
        Size pattern_size = new Size(PAT_COL, PAT_ROW);
        List<Mat> img_points = new List<Mat>();
        int found_num = 0;
        for (i = 0; i < IMAGE_NUM; i++)
        {
            MatOfPoint2f corners = new MatOfPoint2f();
            bool found = false;
            if (USE_FIND_CHESSBOARD_CORNERS_SB_METHOD)
            {
                found = Calib3d.findChessboardCornersSB(srcImages[i], pattern_size, corners, findCornersFlags);
            }
            else
            {
                found = Calib3d.findChessboardCorners(srcImages[i], pattern_size, corners, findCornersFlags);
            }

            if (found)
            {
                Debug.Log(String.Format("{0:00}", i) + "... ok");
                found_num++;
            }
            else
            {
                Debug.Log(String.Format("{0:00}", i) + "... fail");
            }

            if (!USE_FIND_CHESSBOARD_CORNERS_SB_METHOD && ENABLE_CORNER_SUB_PIX)
            {
                using (Mat src_gray = new Mat(srcImages[i].size(), CvType.CV_8UC1))
                {
                    Imgproc.cvtColor(srcImages[i], src_gray, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.cornerSubPix(src_gray, corners, new Size(winSize, winSize), new Size(-1, -1), new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 30, 0.0001));
                }
            }

            Calib3d.drawChessboardCorners(srcImages[i], pattern_size, corners, found);
            img_points.Add(corners);


            using (Mat rgba = new Mat(srcImages[i].size(), CvType.CV_8UC4))
            {
                Imgproc.cvtColor(srcImages[i], rgba, Imgproc.COLOR_BGR2RGBA);
                Utils.fastMatToTexture2D(rgba, texture);
            }
            yield return new WaitForSeconds(0.2f);
        }

        if (found_num != IMAGE_NUM)
        {
            Debug.Log("Calibration Images are insufficient.");
        }


        // Find intrinsic and extrinsic camera parameters.
        double aspect_raito = 1.0;
        int calibrationFlags = calibrateCameraROFlags;
        Mat cam_mat = Mat.eye(3, 3, CvType.CV_64F);
        if ((calibrationFlags & Calib3d.CALIB_FIX_ASPECT_RATIO) == Calib3d.CALIB_FIX_ASPECT_RATIO)
            cam_mat.put(0, 0, new double[] { aspect_raito });
        Mat dist_coefs = Mat.zeros(8, 1, CvType.CV_64F);
        List<Mat> rvecs = new List<Mat>();
        List<Mat> tvecs = new List<Mat>();

        int iFixedPoint = -1;
        if (release_object)
            iFixedPoint = PAT_COL - 1;
        double repErr = Calib3d.calibrateCameraRO(
            obj_points,
            img_points,
            srcImages[0].size(),
            iFixedPoint,
            cam_mat,
            dist_coefs,
            rvecs,
            tvecs,
            newObjPoints,
            calibrationFlags | Calib3d.CALIB_USE_LU
            );

        if (release_object)
        {
            Debug.Log("New board corners: ");
            Point3[] newPoints = new MatOfPoint3f(newObjPoints).toArray();
            Debug.Log(newPoints[0]);
            Debug.Log(newPoints[PAT_COL - 1]);
            Debug.Log(newPoints[PAT_COL * (PAT_ROW - 1)]);
            Debug.Log(newPoints[newPoints.Length - 1]);
        }


        // Print camera parameters to the output file.


        //
        //Debug.Log("obj_points: " + obj_points.Count);
        //Debug.Log("obj_points: " + obj_points[0].dump());
        //Debug.Log("img_points: " + img_points.Count);
        //Debug.Log("img_points: " + img_points[0].dump());
        //Debug.Log("srcImages[0].size(): " + srcImages[0].size());
        Debug.Log("intrinsic: " + cam_mat.dump());
        Debug.Log("distortion: " + dist_coefs.dump());
        Debug.Log("repErr: " + repErr);
        Debug.Log("USE_FIND_CHESSBOARD_CORNERS_SB_METHOD: " + USE_FIND_CHESSBOARD_CORNERS_SB_METHOD);
        Debug.Log("findCornersFlags: " + findCornersFlags);
        Debug.Log("ENABLE_CORNER_SUB_PIX: " + ENABLE_CORNER_SUB_PIX);
        Debug.Log("USE_NEW_CALIBRATION_METHOD: " + USE_NEW_CALIBRATION_METHOD);
        Debug.Log("calibrationFlags: " + calibrationFlags);
        //

        yield return null;
    }
}
