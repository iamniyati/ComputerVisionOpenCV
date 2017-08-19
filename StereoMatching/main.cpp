/**
 * This class calculates the disparity map
 *  and average displacement of pixels
 *  using stereo image processing
 *  in a sequence of images and video.
 *
 * @version main.cpp
 *
 * @author   Niyati Shah
 *
 * @site https://github.com/npinto/opencv/blob/master/samples/cpp/stereo_match.cpp
 *       https://github.com/sourishg/disparity-map/blob/master/epipolar.cpp
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;


// Initializing stereo parameters
int window_size = 9 ;
int number_of_disparities = 16*5;
int pre_filter_size = 7;
int pre_filter_cap = 31;
int min_disparity = 10;
int texture_threshold = 800;
int uniqueness_ratio =10;
int max_diff = 100;
int speckle_window_size = 100;

//Declaring other vairbles
Mat img_left, img_right, img_left_disp, img_right_disp;
Mat img_left_desc, img_right_desc;
vector< KeyPoint > LeftImKeyPt, RightImKeyPt;


//  Function to help user get started with the running of the program.
//  For a sequence of images, it shows the user how to give input
void help() {
    cout << "\nThis program takes the image sequence as a parameter \n"
         << " And uses two images from it."
         << " First is the left image and second is the right image\n"
         << " Once all three images are displayed, click on any point on \n"
         <<  " left image and see the average displacement that is calculated."
         << endl;
}


// Function to check if the current
// point is a left key point feature.
bool isLeftKeyPoint(int i, int j) {
    // get size of left key point
    int n = LeftImKeyPt.size();
    return (i >= LeftImKeyPt[0].pt.x && i <= LeftImKeyPt[n-1].pt.x
            && j >= LeftImKeyPt[0].pt.y && j <= LeftImKeyPt[n-1].pt.y);
}

// Function to check if the current
// point is a right key point feature.
bool isRightKeyPoint(int i, int j) {
    // get size of right key point
    int n = RightImKeyPt.size();
    return (i >= RightImKeyPt[0].pt.x && i <= RightImKeyPt[n-1].pt.x
            && j >= RightImKeyPt[0].pt.y && j <= RightImKeyPt[n-1].pt.y);
}

// Cost function to calculate penalty for
// the correct corresponding point.
long costF(const Mat& left, const Mat& right) {
    long cost = 0;
    // Cost is the summation of the absolute difference
    // between the left and right greyscale images' pixel
    // intensity value
    for (int i = 0; i < 32; i++) {
        cost += abs(left.at<uchar>(0,i)-right.at<uchar>(0,i));
    }
    return cost;
}

// Function to find the corresponding point in the right image
// When a point is selected on the left image
int getCorresPoint(Point p, Mat& img, int ndisp) {
    // Declare all variables needed.
    int window = 5;
    // initialise mincost as one times ten to the ninth power
    long minCost = 1e9;
    // initialise choosen point as 0
    int chosen_i = 0;
    // Initialise the first left and right key points in x and y co-ordinates
    // and last of y  keypoint in left and right keypoints.
    int x0r = RightImKeyPt[0].pt.x;
    int y0r = RightImKeyPt[0].pt.y;
    int x0l = LeftImKeyPt[0].pt.x;
    int y0l = LeftImKeyPt[0].pt.y;
    int ynr = RightImKeyPt[RightImKeyPt.size()-1].pt.y;
    int ynl = LeftImKeyPt[LeftImKeyPt.size()-1].pt.y;

    // Traverse through the image and find the corresponding right point
    for (int pointIdx = p.x-ndisp; pointIdx <= p.x; pointIdx++) {
        long cost = -1;
        // for each x and y  point check for all all weights.
        for (int xweightIdx = -window; xweightIdx <= window; xweightIdx++) {
            for (int yweightIdx = -window; yweightIdx <= window; yweightIdx++) {
                // Check if the point choosen is a left image point of a right image point.
                if (!isLeftKeyPoint(p.x+xweightIdx, p.y+yweightIdx) || !isRightKeyPoint(pointIdx+xweightIdx, p.y+yweightIdx))
                    continue;
                // Get the index of the descriptors.
                int idxl = (p.x+xweightIdx-x0l)*(ynl-y0l+1)+(p.y+yweightIdx-y0l);
                int idxr = (pointIdx+xweightIdx-x0r)*(ynr-y0r+1)+(p.y+yweightIdx-y0r);
                cost += costF(img_left_desc.row(idxl), img_right_desc.row(idxr));
            }
        }
        cost = cost / ((2*window+1)*(2*window+1));
        // If cost is less than min cost then
        // update the minimum cost.
        // and the choosen point.
        if (cost < minCost) {
            minCost = cost;
            chosen_i = pointIdx;
        }
    }
    return chosen_i;
}

// Function that is to be performed when a the user does a left click with mouuse.
void mouseClickLeft(int event, int x, int y, int flags, void* userdata) {
    // check if event if a left click
    if (event == EVENT_LBUTTONDOWN) {
        // check if the left click is on the left image.
        if (!isLeftKeyPoint(x,y))
            return;
        // find the right corresponding point to the left clicked point.
        int right_i = getCorresPoint(Point(x,y), img_right, 20);
        // Display the points and the  distance between them.
        cout << "Left: X:" << x <<" Y: "<<y << " Right: X:" << right_i <<" Y: "<<y<< endl;
        cout<< "Distance of selected point between both the images is: "<<x-right_i<<" pixels";
    }
}

// Function to find feature points in the images
// using the ORB feature point detector
void cacheDescriptorVals() {
    // Initialise a descriptor extractor as ORB detector
    Ptr<DescriptorExtractor> extractor = ORB::create();

    // FOr each point in image look for key points
    // Where the diameter is kept as 1.
    for (int y = 0; y < img_left.cols; y++) {
        for (int x = 0; x < img_left.rows; x++) {
            LeftImKeyPt.push_back(KeyPoint(y,x,1));
            RightImKeyPt.push_back(KeyPoint(y,x,1));
        }
    }
    // compute if its a keypoint or not.
    extractor->compute(img_left, LeftImKeyPt, img_left_desc);
    extractor->compute(img_right, RightImKeyPt, img_right_desc);
}

int main(int argc, char *argv[]) {

//     If the path of the folder with images is not provided, prompt the user
//     to input the folder path and exit the program.
    if (argc != 2) {
        help();
        return 1;
    }

    // This part calculates the calibratiom. Can be uncommented
    // but takes very long to process all the pictures to find calibration
    // So using precomputed calibration that is stored in intrinsicsPartC.xml file.
/*
       int numBoards;
       int numCornersHor;
       int numCornersVer;
       Size imageSize;

       numBoards = 25;
       numCornersHor = 6;
       numCornersVer = 8;
       Mat image,GrayImage, cornerImage;
       // Calculate the number of squares and board size
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
   ////        drawChessboardCorners(image, board_sz, corners, found);
   ////        // Resize and display the image with the found corners.
   ////        resize(image,cornerImage,Size(640,480));
   ////        imshow("Part A: find corners", cornerImage);
   //
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
       FileStorage fs("intrinsicsPartC.xml", FileStorage::WRITE);
       fs << "image_width" << imageSize.width << "image_height" << imageSize.height
          << "camera_matrix" << intrinsic << "distortion_coefficients"
          << distCoeffs;
       fs.release();
       capture.release();


        capture =VideoCapture(argv[1]);
       if (!capture.isOpened()) {
           cout << "\nVideoCapture not initialized properly\n";
           return -1;
       }
*/

    // Reinitialise the capture to get all images.
    // comment this code if above code is uncommented
    ///////////////////////////
    Mat image;
    FileStorage fs;
    VideoCapture capture = VideoCapture(argv[1]);
    capture >> image;
    ////////////////////////////


    fs.open("intrinsicsPartC.xml", FileStorage::READ);
    cout << "\nimage width: " << (int) fs["image_width"];
    cout << "\nimage height: " << (int) fs["image_height"];
    Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distortion_coeffs_loaded;
    cout << "\nintrinsic matrix:" << intrinsic_matrix_loaded;
    cout << "\ndistortion coefficients: " << distortion_coeffs_loaded << endl;
    Mat map1, map2;

    cout << "*****************Caliberation Calculation Complete********************";

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
    // Declare variables needed for processing.
    Mat LeftImage, RightImage, LeftGreY, RightGrey;
    Ptr<StereoBM> sbm = StereoBM::create(16, 15);
    int countFrames = 0;


    // Look through the images and find the correct image.
    // and recalibrate/remap the images.
    do{
        Mat imageUndistorted ;
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
        // Choosing only images that are two images apart.
        if (countFrames==0) {
            LeftImage = imageUndistorted;
        }
        if (countFrames==2) {
            RightImage = imageUndistorted;

        }
        countFrames++;
        if ((waitKey(30)) == 27) break;
        capture >> image;
    }while(!image.empty());


    cout<<"\nFound Necessary LEft and Right Images\n";
    // Resizing the images and converting them to greyscale
    resize(LeftImage, LeftImage, Size(640, 480));
    resize(RightImage, RightImage, Size(640, 480));
    cvtColor(LeftImage, LeftGreY, CV_BGR2GRAY);
    cvtColor(RightImage, RightGrey, CV_BGR2GRAY);

    // Calibrating and tuning the variables required to create the disparity map
    Mat disp, disp8;
    Rect roi1, roi2;
    sbm->setBlockSize(window_size); // size of averaging window used to match pixel blocks (odd number between 5 and 255)
    sbm->setNumDisparities(number_of_disparities); //size of disparity range () multiple of 16

    sbm->setPreFilterSize(pre_filter_size);  // size of block used in PREFILTER_NORMALIZED_RESPONSE mode
    min_disparity = -min_disparity;
    sbm->setMinDisparity(min_disparity); // the minimum disparity, usually 0
    sbm->setTextureThreshold(texture_threshold); // textureness threshold,
    sbm->setUniquenessRatio(uniqueness_ratio); // Uniqueness threshold defines what it means to be a
    // "clear winner," the margin in %% between the best and second-best.
    max_diff = 0.01 * ((float) max_diff);
    sbm->setDisp12MaxDiff(max_diff);
    sbm->setSpeckleRange(32); //8  //  allowed difference between neighbor pixels
    sbm->setSpeckleWindowSize(speckle_window_size); // maximum size of a speckle to be considered as speckle
    sbm->setPreFilterCap(pre_filter_cap);  // saturation threshold applied after pre-filtering
    sbm->setROI1(roi1); //Region of interest rectagles
    sbm->setROI2(roi2); //Region of interest rectagles

    // COmpute the stereo disparity between the left and right image
    sbm->compute(LeftGreY, RightGrey, disp);
    // Convert the image to 8bit unsigned image
    // to see the actual change from expanded range.
    normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
    imshow("disp", disp8);

    // Second PArt of Part C
    // Declare and initialise variables to calculate avergae displacement.
    img_left = LeftImage;
    img_right = RightImage;
    img_left_disp =LeftImage;
    img_right_disp = RightImage;
    // Find the key descriptors using the ORB detector
    cacheDescriptorVals();
    namedWindow("IMG-LEFT", 1);
    namedWindow("IMG-RIGHT", 1);

    // Display the image and wait till the user clicks
    // left click om left image to find the average displacement
    setMouseCallback("IMG-LEFT", mouseClickLeft, NULL);
    while (1) {
        imshow("IMG-LEFT", img_left_disp);
        imshow("IMG-RIGHT", img_right_disp);
        if (waitKey(30) > 0) {
            break;
        }
    }
    waitKey(0);
    return(0);
}