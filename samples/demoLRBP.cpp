#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/// Global variables
char* source_window = "Source image";
char* rotate_window = "Rotate";

/** @function main */
int main( int argc, char** argv )
{
  Point2f srcTri[3];
  Point2f dstTri[3];
  
  Mat rot_mat( 2, 3, CV_32FC1 );
  Mat src, rotate_dst;
  
  /// Load the image
  src = imread( argv[1], 1 );
    
  /** Rotating the image after Warp */
  
  /// Compute a rotation matrix with respect to the center of the image
  Point center = Point( src.cols/2, src.rows/2 );
  double angle = 0.0;
  double scale = 1;
  double diag = sqrt(src.rows*src.rows+src.cols*src.cols);
  
  /// Get the rotation matrix with the specifications above
  rot_mat = getRotationMatrix2D( center, angle, scale );
  rot_mat.at<double>(0,2) += (diag - src.cols)/2.0;
  rot_mat.at<double>(1,2) += (diag - src.rows)/2.0;
  
  Size size = Size(diag,diag);
  
  /// Rotate the warped image
  warpAffine( src, rotate_dst, rot_mat, size );
  
  /// Show what you got
  namedWindow( source_window, CV_WINDOW_AUTOSIZE );
  imshow( source_window, src );
  
  namedWindow( rotate_window, CV_WINDOW_AUTOSIZE );
  imshow( rotate_window, rotate_dst );
  
  while( true )
  {
    int c = waitKey(500);
    /// Press 'ESC' to exit the program
    if( (char)c == 27 )
    { break; }
    
    angle+=5;
    /// Get the rotation matrix with the specifications above
    rot_mat = getRotationMatrix2D( center, angle, scale );
    rot_mat.at<double>(0,2) += (diag - src.cols)/2.0;
    rot_mat.at<double>(1,2) += (diag - src.rows)/2.0;
  
    Size size = Size(diag,diag);
  
    /// Rotate the warped image
    warpAffine( src, rotate_dst, rot_mat, size );

    imshow( rotate_window, rotate_dst );

  }
  
  /// Wait until user exits the program
  //waitKey(0);
  
  return 0;
}
