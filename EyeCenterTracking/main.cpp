/**
 * HW06
 * Main class to apply the Timm's algorithm
 * to detect eye center using means of gradient
 *
 *
 * @version   main.cpp
 *
 * @author   Niyati Shah
 *
 * Cite: For BioID images: https://www.bioid.com/About/BioID-Face-Database
 * Cite: For HAAR Cascades: opencv/data/haarcascades/
 * Cite: For code: https://github.com/trishume/eyeLike
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;


// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 0.3;;

// Postprocessing
const float kPostProcessThreshold = 0.97;


/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string main_window_name = "Capture - Face detection";
string face_window_name = "Capture - Face";
Mat debugImage;



/**
* @function matrixMagnitude:  Calcualye the magnitude of matrix
 *                              using the x and y gradient
*
* matX: X gradient
 * maty : Y gradient
*/
Mat matrixMagnitude(const Mat &matX, const Mat &matY)
{
    // Initialise the magnitude mat object
    Mat mags(matX.rows, matX.cols, CV_64F);
    //For each row and column on image
    for (int y = 0; y < matX.rows; ++y)
    {
        //Get the pointer to current row in input and output image
        const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x)
        {
            // Calculaye the magnitude and store in output image
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}


/**
* @function computeDynamicThreshold:  Compute the dynamic threshold
*                   using standard deviation
* mat: given mat image
 * stdDevFactor: standard deviation factor
*/
double computeDynamicThreshold(const Mat &mat, double stdDevFactor)
{
    // create the variable for computing standard deviation
    Scalar stdMagnGrad, meanMagnGrad;
    // Perfrom the mean standard deviation on the image
    meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}


/**
* @function testPossibleCentersFormula: the papers mathematical algorihtm to
 *                 to check if the given point is a poissile center
*
 * x: x co-ordinate
 * y: y-cordinate
* weight: given weigted mat image
 * gradX: gradient of x
 * gradY: gradient of y
 * out: output image
*/
void testPossibleCentersFormula(int x, int y, const Mat &weight, double gradX, double gradY, Mat &out)
{
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy)
    {
        //Get the pointer to current row in input and output image
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);

        for (int cx = 0; cx < out.cols; ++cx)
        {
            if (x == cx && y == cy)
            {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize the displacement vector
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gradX + dy*gradY;
            // Make all negative dot products as zero
            dotProduct = max(0.0, dotProduct);
            // square and multiply by the weight
             Or[cx] += dotProduct * dotProduct * (Wr[cx] / kWeightDivisor);

        }
    }
}

/**
* @function unscalePoint:  Rescale the given point
*
* mat: given mat image
*/
Point unscalePoint(Point p, Rect origSize)
{
    float ratio = (((float)kFastEyeWidth) / origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return Point(x, y);
}


/**
* @function computeMatGradient:  COmpute the gradient
*                         of the given image
* mat: given mat image
*/
Mat computeMatGradient(const Mat &mat)
{
    // Create an output image of input image size
    Mat outIm(mat.rows, mat.cols, CV_64F);
    // For each row of image
    for (int row = 0; row < mat.rows; ++row)
    {
        //Get the pointer to current row in input and output image
        const uchar *Mr = mat.ptr<uchar>(row);
        double *Or = outIm.ptr<double>(row);

        Or[0] = Mr[1] - Mr[0];
        // For each column of image
        for (int col = 1; col < mat.cols - 1; ++col)
        {
            Or[col] = (Mr[col + 1] - Mr[col - 1]) / 2.0;
        }
        Or[mat.cols - 1] = Mr[mat.cols - 1] - Mr[mat.cols - 2];
    }

    return outIm;
}


/**
* @function floodShouldPushPoint:  check if the given point is within
*                         the matrix (defined rows and columns)
* mat: given image
* newPt: given point
*/
bool floodShouldPushPoint(const Point &newPt, const Mat &mat)
{
    return newPt.x >= 0 && newPt.x < mat.cols && newPt.y >= 0 && newPt.y < mat.rows;
}

/**
* @function floodKillEdges: remove the unwnated edges and
*                        return the mask
* mat: given image
*
*/
Mat floodKillEdges(Mat &mat)
{
    // Create a rectangle of given image size
    rectangle(mat, Rect(0, 0, mat.cols, mat.rows), 255);
    // Create a mask mat object of input mat size
    Mat mask(mat.rows, mat.cols, CV_8U, 255);
    // Create a queue and initialise it to point 0,0
    queue<Point> toDo;
    toDo.push(Point(0, 0));

    while (!toDo.empty())
    {
        // get a point from the quete
        Point pt = toDo.front();
        toDo.pop();
        // Ignore if the point is zer0
        if (mat.at<float>(pt) == 0.0f)
        {
            continue;
        }
        // add a new in every direction
        Point newPt(pt.x + 1, pt.y); // right
        if (floodShouldPushPoint(newPt, mat)) toDo.push(newPt);
        newPt.x = pt.x - 1;
        newPt.y = pt.y; // left
        if (floodShouldPushPoint(newPt, mat)) toDo.push(newPt);
        newPt.x = pt.x;
        newPt.y = pt.y + 1; // down
        if (floodShouldPushPoint(newPt, mat)) toDo.push(newPt);
        newPt.x = pt.x;
        newPt.y = pt.y - 1; // up
        if (floodShouldPushPoint(newPt, mat)) toDo.push(newPt);
        // change the pixel at that point to 0 (or kill it)
        mat.at<float>(pt) = 0.0f;
        mask.at<uchar>(pt) = 0;
    }
    return mask;
}

/**
* @function scaleToFastSize : resize the given image
*
* src: the source Mat image
* dst: resized output image
*/
void scaleToFastSize(const Mat &src, Mat &dst)
{
    resize(src, dst, Size(kFastEyeWidth, (((float)kFastEyeWidth) / src.cols) * src.rows));
}

/**
* @function findEyeCenter : Search for center of the eyes in
*                        the detected eye in the face detected in the image
* face: given frame
* eye: Rectangle where the eye is detected
*/
Point findEyeCenter(Mat face, Rect eye, string debugWindow)
{
    // Initialize the required variables
    Mat eyeROIUnscaled = face(eye);
    Mat eyeROI;
    // resize the given image
    scaleToFastSize(eyeROIUnscaled, eyeROI);
    // draw eye region
    rectangle(face, eye, 1234);

    // Compute the x and y gradient using the current and transposed mat object
    // containing the eye
    Mat gradientX = computeMatGradient(eyeROI);
    Mat gradientY = computeMatGradient(eyeROI.t()).t();

    // Normalize and threshold the gradient
    // compute all the magnitudes
    Mat mags = matrixMagnitude(gradientX, gradientY);
    //compute the threshold
    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);

    //normalize the gradient.
    for (int y = 0; y < eyeROI.rows; ++y)
    {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < eyeROI.cols; ++x)
        {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > gradientThresh)
            {
                Xr[x] = gX / magnitude;
                Yr[x] = gY / magnitude;
            }
            else
            {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }
    imshow(debugWindow, gradientX);

    // Create a blurred and inverted image for weighting
    Mat weight;
    GaussianBlur(eyeROI, weight, Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
    for (int y = 0; y < weight.rows; ++y)
    {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x)
        {
            row[x] = (255 - row[x]);
        }
    }

    // Initialise a zeros image of the size of eye
    Mat outSum = Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);
    // for each possible gradient location

    // For each row and column of weighted image
    // check if it could be the eye center
    for (int wRows = 0; wRows < weight.rows; ++wRows)
    {
        const double *Xr = gradientX.ptr<double>(wRows);
        const double *Yr = gradientY.ptr<double>(wRows);
        for (int wCols = 0; wCols < weight.cols; ++wCols)
        {
            double gradX = Xr[wCols], gradY = Yr[wCols];
            // If gradient are zero ignore
            if (gradX == 0.0 && gradY == 0.0)
            {
                continue;
            }
            // Check if given point is a possible center or not
            testPossibleCentersFormula(wCols, wRows, weight, gradX, gradY, outSum);
        }
    }
    // scale all the values down by averaging
    double numGradients = (weight.rows*weight.cols);
    Mat out;
    outSum.convertTo(out, CV_32F, 1.0 / numGradients);
    // Find the maximum point
    Point maxP;
    double maxVal;
    minMaxLoc(out, NULL, &maxVal, NULL, &maxP);
    //Apply a Flood fill to the edges
    Mat floodClone;
    double floodThresh = maxVal * kPostProcessThreshold;
    threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
    Mat mask = floodKillEdges(floodClone);
    minMaxLoc(out, NULL, &maxVal, NULL, &maxP, mask);

    return unscalePoint(maxP, eye);
}


/**
* @function findEyes : Search for eyes in
*                        the detected face in the image
* frame_gray: Grayscale frame
        * Rect: Rectangle where the face is detected
*/
void findEyes(Mat frame_gray, Rect face)
{
    // Initialise the varibles.
    Mat faceROI = frame_gray(face);
    Mat debugFace = faceROI;


    // Find eye regions and draw them using the pre defined constants
    // Here we know that given a size of a face, we can accurately
    // assume the size if the eye and where it could be located.
    int eye_region_width = face.width * (kEyePercentWidth / 100.0);
    int eye_region_height = face.width * (kEyePercentHeight / 100.0);
    int eye_region_top = face.height * (kEyePercentTop / 100.0);

    Rect leftEyeRegion(face.width*(kEyePercentSide / 100.0),
                       eye_region_top, eye_region_width, eye_region_height);
    Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide / 100.0),
                        eye_region_top, eye_region_width, eye_region_height);

    // Find Eye Centers
    Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye");
    Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye");

    // get corner regions using the center if the eye.

    // Get the left eye's right corner region
    Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;

    // Get the left eye's left corner region
    Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;

    // Get the right eye's left corner region
    Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;

    // Get the right eye's right corner region
    Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;

    rectangle(debugFace, leftRightCornerRegion, 200);
    rectangle(debugFace, leftLeftCornerRegion, 200);
    rectangle(debugFace, rightLeftCornerRegion, 200);
    rectangle(debugFace, rightRightCornerRegion, 200);

    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;

    // draw eye centers
    circle(debugFace, rightPupil, 3, 1234);
    circle(debugFace, leftPupil, 3, 1234);
    
    imshow(face_window_name, faceROI);
}



/**
 * @function detectAndDisplay : Display the read image
 *                              And detect a face in the image
 * frame: Image where theface will be detected
*/
void detectAndDisplay(Mat frame)
{
    // Create variables to store data
    vector<Rect> faces;          // Rectangle to store ROI ie region of Face
    vector<Mat> rgbChannels(3);  // Store the seperate channels of image
    split(frame, rgbChannels);   // Split the frames
    Mat frame_gray = rgbChannels[2]; // Get grayscale frame

    // Using HAAR Cascades to detetc the face in the given image/frame
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT, Size(150, 150));

    // Create a rectangle around
    for (int i = 0; i < faces.size(); i++)
    {
        rectangle(frame, faces[i], (255,255,0));
    }

    // If a face is detected, then search for the eyes in the first detected face
    if (faces.size() > 0)
    {
        findEyes(frame_gray, faces[0]);
    }
}


//  Function to help user get started with the running of the program.
//  For a sequence of images, it shows the user how to give input
void help() {
    cout << "\nThis program gets you started reading a sequence of images "
         << "\n or using the inbuilt camera"
         << "\n If nothing is passed the inbuilt camera is started and the program continues\n"
         << " For an image sequence: "
         << " \n  Given path of images in folder: BioID-FaceDatabase-V1\n"
         << "   Usage:   <path to the first image in the sequence>\n"
         << "   Example:/abc/HW5/BioID-FaceDatabase-V1/%4d.pgm\n"
         << " To quit : Press q"
         << endl;
}

/**
* @function main
*/
int main(int argc, const char** argv)
{

    help();

    VideoCapture capture;
    Mat frame;

    // Load the cascades
    if (!face_cascade.load(face_cascade_name))
    {
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    };

    namedWindow(main_window_name, CV_WINDOW_NORMAL);
    namedWindow(face_window_name, CV_WINDOW_NORMAL);
    namedWindow("Right Eye", CV_WINDOW_NORMAL);
    namedWindow("Left Eye", CV_WINDOW_NORMAL);


    // If an input if given use that video else start the system camera
    if (argc > 1)
        capture.open(argv[1]);
    else
        capture.open(0);

    // Set the FPS be 20
    capture.set(CV_CAP_PROP_FPS, 20);

    // Check if he video is correctly binded with the class
    // Else exit the program.
    int i=0;
    if (capture.isOpened())
    {
        while (true)
        {

            capture >> frame;

            frame.copyTo(debugImage);

            // Apply the classifier to the frame
            if (!frame.empty())
            {
                detectAndDisplay(frame);
            }
            else
            {
                cout << "Video ended or frame could not be read "<< endl;
                return -3;
            }

            imshow(main_window_name, frame);

            // Press q to exit the program
            int c = waitKey(10);
            if ((char)c == 'q')
            {
                break;
            }


        }
    }else{
        cout << "Cannot open the video file" << endl;
        return -1;
    }


    return 0;
}

