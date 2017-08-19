
/**
 * This class is used to apply the watershed algorithm on a
 * list of images to perform segmentation
 *
 * @version   $Id$ 1.0 HW02_Shah_Niyati.cpp
 *
 * @author   Niyati Shah
 *
 * Cite:
 *	$Log$
 */


#include <iostream>
#include <opencv2/opencv.hpp>

// Defining the context of cv and std to avoid using cv and std as prefixes.
using namespace cv;
using namespace std;

// initialise a masker and an image Mat object,
// and a point that would store the previous point
// which is initialised to -1, -1, which is outside
// image.

Mat markerMask, OrigImage;
Point prevPt(-1, -1);

/**
 *  This function is used to help the user as to how this program works
 */
static void help() {
    cout <<"\n ************ HELP DOC ***************\n";
    cout << "\nThis program shows how a supervised watershed segmentation algorithm works\n";

    cout << "Directions : \n"
            "\tESC - quit the program\n"
            "\tr - restore the original image\n"
            "\tw or SPACE - run watershed segmentation algorithm\n"
            "\tSelect Segment: Click and drag the mouse on the image to select a\n"
            "\t\t part of the image and leave once the part desired is selected.\n"
            "\t\t Repeat till all parts required are selected \n"
            "\t\t Select 'r' to restore the original image if any mistake is made\n"
            "\t\t or select 'w' or SPACE to apply watershed transform on the image\n\n";

    cout << " After the watershed algorithm is run, press ESC to quite \n"
            "OR \n"
            " you can select more segments on the original image as given in directions\n "
            " and see more changes using the watershed algorithm";
    cout <<"\n ************ END OF DOC ***************\n";
}

/**
 *
 * This function is for the on mouse events which is used to
 * mark the different segments of the image for it to perform
 * supervised learning.
 *
 * @param event : integer for the type of mouse event
 * @param x : x coordinate of the mouse event
 * @param y : y coordinate of the mouse event
 * @param flags : condition whenever a mouse event occurs
 */
static void onMouse( int event, int x, int y, int flags, void* )
{
    // if mouse event is not within the borders of given image then no action taken
    if( x < 0 || x >= OrigImage.cols || y < 0 || y >= OrigImage.rows )
        return;

    // check if the left mouse button released and set presvious point to (-1,-1)
    if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
        prevPt = Point(-1,-1);
    else if( event == EVENT_LBUTTONDOWN ) // if the left button is clicked (no dragging)
        prevPt = Point(x,y);
    else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) ) // if there is a dragging
    {
        // create a point x,y
        Point pt(x, y);
        // check if the x co-ordinate is less than 0
        // then previous point becomes current point
        if( prevPt.x < 0 )
            prevPt = pt;
        // on the mask image and given image , make a line
        // using the point co-ordinates
        // of the current point and previous point
        line( markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        line( OrigImage, prevPt, pt, Scalar::all(255), 5, 8, 0 );
        // store the current point in the previous point
        prevPt = pt;
        // display the image with the changes
        imshow("image", OrigImage);
    }

}

int main( int argc, char** argv )
{
    // Check if the filename is passed to the programme
    if ( argc != 2 )
    {
        cout << "Error: Too few arguments, include image file name" << endl;
        return -2;

    }
    else {

        // call the help function to learn about how to run the program
        help();

        // Read the image.
        Mat ReadImage = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

        // Create two mat object
        // one to store the grayscale image
        // and a temporary image that is
        // initialised by temporary image
        Mat  imgGray, TempImg = ReadImage;

        // If there is an error opening the image then throw an error
        if (TempImg.empty()) {
            cout << "Couldn't open image " << argv[1] << "\n";
            return 0;
        }

        //create a window with a name to display the image in a fixed (no resize allowed) size format.
        namedWindow("image", WINDOW_AUTOSIZE);

        // Copies the temporary image to a original image
        TempImg.copyTo(OrigImage);

        // Convert the image to gray scale and store in makerMask mat object
        cvtColor(OrigImage, markerMask, COLOR_BGR2GRAY);

        // To see the overlap of the segments over the original image
        //  the below command is used.
        cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);


        // fill zeroes in a markerMask Mat object containing image
        markerMask = Scalar::all(0);


        // Display the given image
        imshow("image", OrigImage);

        //  Check for mouse click on image
        setMouseCallback("image", onMouse, 0);


        //
        // infinite for loop to keep waiting for the user to
        // select the appropriate option as described in the
        // help function
        //
        while(true){

            // Use a waitkey to keep displaying the image till the user interrupts it.
            int c = waitKey(0);
            if ((char) c == 27)
                break;

            // If a mistake is made while selecting segments
            // click 'r' to clear the image and restart
            // selection of segments.
            if ((char) c == 'r') {
                // fill zeroes in a markerMask Mat object
                markerMask = Scalar::all(0);
                // copy the image
                TempImg.copyTo(OrigImage);
                // display  the image
                imshow("image", OrigImage);
            }

            // If 'w' or space is selected
            // start the watershed
            if ((char) c == 'w' || (char) c == ' ') {

                // initialise a segment count to 0
                // create a vector of vector to store multiple contours
                // create a 4dimensional vector to store hierarchy of contours
                int compCount = 0;
                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;

                //  find contours in the image
                // use the RETR_CCOMP to get all contours and put them into a two-level hierarchy
                // use the CHAIN_APPROX_SIMPLE to compress horizontal, vertical, and diagonal segments
                // into only their endpoints
                findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

                // check if atleast one segment is selected
                if (!contours.empty()) {

                    // Create a mat object markers that is of size of the masker and type CV_32S
                    // and fill the fill zeroes in
                    Mat markers(markerMask.size(), CV_32S);
                    markers = Scalar::all(0);

                    // For each of the contour that is found earlier, draw the shape
                    // and also count the number of contours made,
                    for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++) {
                        drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
                    }

                    // Create a vector object to store the
                    // different colors that are randomly generated
                    vector<Vec3b> colorTab;

                    // generate a random colors for each contour
                    // using the random number generator fucntion
                    // between 0 and 255
                    for (int idx = 0; idx < compCount; idx++) {
                        int blue = theRNG().uniform(0, 255);
                        int green = theRNG().uniform(0, 255);
                        int red = theRNG().uniform(0, 255);
                        colorTab.push_back(Vec3b((uchar) blue, (uchar) green, (uchar) red));
                    }

                    // run the watershed function on the image and mask
                    watershed(TempImg, markers);

                    // create a mat object to store output image
                    Mat outputImage(markers.size(), CV_8UC3);

                    // paint through the contours of the watershed
                    // image by going through each row and column
                    for (int row = 0; row < markers.rows; row++) {
                        for (int col = 0; col < markers.cols; col++) {
                            int index = markers.at<int>(row, col);
                            if (index == -1)
                                outputImage.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
                            else if (index <= 0 || index > compCount)
                                outputImage.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                            else
                                outputImage.at<Vec3b>(row, col) = colorTab[index - 1];
                        }
                    }

                    // superimpose the watershed transform image
                    // with greyscale of original image
                    outputImage = outputImage * 0.6 + imgGray * 0.3;

                    //Display the image with the watershed transform on it.
                    imshow("watershed transform", outputImage);
                    

                }else{
                    // Error message if no segments selected
                    cout<<"\n\nPlease select at least one segment to form contours\n"
                            " OR Look at help doc to see how to select segments ";
                }
            }
        }
    }
    return 0;
}