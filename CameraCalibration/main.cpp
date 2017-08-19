/**
 * This class calculates caliberation correction
 * from a sequence of images, and applies
 * the caliberation to rectify the iamge.
 *
 * @version main.cpp
 *
 * @author   Niyati Shah
 *
 * @site http://aishack.in/tutorials/calibrating-undistorting-opencv-oh-yeah/
 *          and  OpenCV- Texbook
 */
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//  Function to help user get started with the running of the program.
//  For a sequence of images, it shows the user how to give input
void help() {
    cout << "\nThis program gets you started reading a sequence of images \n"
         << " For an image sequence: \n"
         << "   Usage:   <path to the first image in the sequence>\n"
         << "   Example:/abc/HW5/%2d.jpg\n"
         << endl;
}


int main(int argc, char *argv[]) {

    // Check if the number of arguments are correct
    // else show the help document and quit.
    if (argc != 2) {
        help();
        exit(-1);
    }
    cout<<"*****************Part A********************";
    // Create and Initialise the the number of boards,
    // number of corners in the width and height.
    int numBoards;
    int numCornersHor;
    int numCornersVer;
    Size imageSize;
    Mat image,GrayImage, cornerImage;

//    printf("Enter number of corners along width: ");
//    scanf("%d", &numCornersHor);
//
//    printf("Enter number of corners along height: ");
//    scanf("%d", &numCornersVer);
//
//    printf("Enter number of boards: ");
//    scanf("%d", &numBoards);
    numBoards = 31;
    numCornersHor = 6;
    numCornersVer = 8;

    // Calculate the numbr of squares and board size
    int numSquares = numCornersHor * numCornersVer;
    Size board_sz = Size(numCornersHor, numCornersVer);

    // Create image and object points to store 2d and 3d points of corners
    // respectively, object co-cordinates and corners of checkers.
    vector<vector<Point3f>> object_points;
    vector<vector<Point2f>> image_points;
    vector<Point3f> objCord;
    vector<Point2f> corners;


    int successes=0;

    // Initialize a video capture object and check if it has been
    // opened correctly or not.
    VideoCapture capture(argv[1]);
    if (!capture.isOpened()) {
        cout << "\nVideoCapture not initialized properly\n";
        return -1;
    }
    capture >> image;
    imageSize = image.size();
    // Creates a list of coordinates of unit space that
    // can hold the corners.
    for(int j=0;j<numSquares;j++)
        objCord.push_back(Point3f(j/numCornersHor, j%numCornersHor, 0.0f));

    // Now loop through the images and find chessboard corners
    while(successes<numBoards) {

        // Check if the image is present or not.
        if (image.empty()) {
            cerr << "Failed to open Image or Video Sequence!\n" << endl;
            return -1;
        }

        // Convert the given image into greyscale image
        cvtColor(image, GrayImage, CV_BGR2GRAY);

        // Call findChessboardCorners to find the corners in the given image.
        // Here you can call CALIB_CB_NORMALIZE_IMAGE -> the image brightness is equalised using equalizeHist()
        // CV_CALIB_CB_ADAPTIVE_THRESH -> the image brightness is equalised using adaptive threshold
        // CALIB_CV_FAST_CHECK --> gave me an error of no such commond
        // CALIB_CB_FILTER_QUADS -->locates the quadrangles resulting from the
        // perspective view of the black squares on the chessboard
        bool found = findChessboardCorners(image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS);

        // If they are found find the accurate location of the corner that are found
        if(found) {
            cornerSubPix(GrayImage, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

            // Store the image corners and object corners in the vectors
            // and increase the success count..
            image_points.push_back(corners);
            object_points.push_back(objCord);
            successes++;

            if(successes>=numBoards)
                break;
        }
        drawChessboardCorners(image, board_sz, corners, found);
        // Resize and display the image with the found corners.
        resize(image,cornerImage,Size(640,480));
        imshow("Part A: find corners", cornerImage);

        // Capture the next image
        capture >> image;

        if ((waitKey(30)) == 27) break;

    }

    // Create Variables to store intrinsic, distortion coefficients.
    // rotation and translation vectors.
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;

    //Initialise the aspect ratiro of the camera as 1.
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;

    // Call the caliberation function to caliberate the camera.
    calibrateCamera(object_points, // K vecs (N pts each, object frame)
                    image_points, // K vecs (N pts each, image frame)
                    image.size(), // Size of input images (pixels)
                    intrinsic, // Resulting 3-by-3 camera matrix
                    distCoeffs,// Vector of 4, 5, or 8 coefficients
                    rvecs,  // Vector of K rotation vectors
                    tvecs, // Vector of K translation vectors
                     0, // Flags control calibration options
                    TermCriteria(
                            TermCriteria::COUNT | TermCriteria::EPS,
                            30, // ...after this many iterations
                            DBL_EPSILON // ...at this total reprojection error
                    )
    );

    // Store the caliberated intrinsic, distortion coefficients.
    // rotation and translation vectors.
    FileStorage fs("intrinsics.xml", FileStorage::WRITE);
    fs << "image_width" << imageSize.width << "image_height" << imageSize.height
       << "camera_matrix" << intrinsic << "distortion_coefficients"
       << distCoeffs;
    fs.release();

    fs.open("intrinsics.xml", FileStorage::READ);
    cout << "\nimage width: " << (int) fs["image_width"];
    cout << "\nimage height: " << (int) fs["image_height"];
    Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;
    Mat map1, map2;
    capture.release();
    cout<<"*****************Part B********************";

    // Calculate the undistorted maps using the camera caliberation
    // parameters.
    initUndistortRectifyMap(
            intrinsic_matrix_loaded,  // 3-by-3 camera matrix
            distortion_coeffs_loaded, // Vector of 4, 5, or 8 coefficients
            cv::Mat(),                // Rectification transformation
            intrinsic_matrix_loaded,  // New camera matrix (3-by-3)
            image.size(),             // Undistorted image size
            CV_16SC2,                 // 'map1' type: 16SC2, 32FC1, or 32FC2
            map1,                     // First output map
            map2                      // Second output map
    );


    // Reinitialise the capture to get all images.
    capture =VideoCapture(argv[1]);
    if (!capture.isOpened()) {
        cout << "\nVideoCapture not initialized properly\n";
        return -1;
    }
    
    capture >> image;
    do {
        Mat imageUndistorted,absDiff;

        // After computing undistortion maps, apply them to the input images.

        remap(
                image,             // Input Image
                imageUndistorted,  // Output undistorted image
                map1,              // First output map
                map2,              // Second output map
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar()
        );

        // Calculate the absolute difference between the input image
        // and the caliberated image.
        absdiff(image, imageUndistorted, absDiff);

        // Resize the image.
        resize(image,image,Size(640,480));
        resize(imageUndistorted,imageUndistorted,Size(640,480));
        resize(absDiff,absDiff,Size(640,480));

        // Display the image.1
        imshow("original Image", image);
        imshow("Undistorted Image", imageUndistorted);
        imshow("Absolute Difference", absDiff);

        if ((waitKey(30)) == 27) break;
        capture >> image;
    }while(!image.empty());
    waitKey(0);
    capture.release();

    return 0;
}

