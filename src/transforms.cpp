#include "transforms.hpp"

Mat radonTransform(Mat src){
  
  Mat rot_mat( 2, 3, CV_32FC1 );
  Mat rotate_dst;
  
  /// Compute a rotation matrix with respect to the center of the image
  Point center = Point( src.cols/2, src.rows/2 );
  double angle = 0.0;
  double scale = 1;
  int diag = ceil(sqrt(src.rows*src.rows+src.cols*src.cols));
  
  Mat radon = Mat::zeros(diag, 180, CV_32F);
  Size size = Size(diag, diag);
  
  /// Rotate the warped image
  warpAffine( src, rotate_dst, rot_mat, size );
  
  for(angle=0; angle<180; angle++)
  {
    /// Get the rotation matrix with the specifications above
    rot_mat = getRotationMatrix2D( center, 90-angle, scale );
    rot_mat.at<double>(0,2) += (diag - src.cols)/2.0;
    rot_mat.at<double>(1,2) += (diag - src.rows)/2.0;
    
    /// Rotate the warped image
    warpAffine( src, rotate_dst, rot_mat, size );
    
    for(int i=0; i<diag; i++)
      for(int j=0; j<diag; j++){
	radon.at<float>(diag-i,angle)+=rotate_dst.at<uint>(i,j);
      }
  }
  
  normalize(radon.clone(), radon, 0, 255, NORM_MINMAX, CV_8U);
  
  return radon;
}