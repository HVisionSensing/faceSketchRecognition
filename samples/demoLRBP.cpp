#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

/** @function main */
int main( int argc, char** argv )
{
  
  vector<Mat> photos, sketches, extra;
  
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  int histSize = 32;
  bool uniform = true; bool accumulate = false;
  
  loadImages(argv[1],photos,1);
  loadImages(argv[2],sketches,1);
  //loadImages(argv[3],extra,1);
 
  int nSketches = sketches.size();
  int nPhotos = photos.size();
  int nExtra = extra.size();
  
  vector<vector<Mat>> photosHist(nPhotos), sketchesHist(nSketches), extraHist(nExtra);
  
  cout << "calc photos" << endl;
  
  for(int ii=0; ii<photos.size(); ii++){
    Mat img = photos[ii], temp, radon, lrbp, hist;
    int w = img.cols, h=img.rows;
    for(int jj=0; jj<3; jj++){
      Mat result;
      int deltaw = w/pow(2,jj), deltah = h/pow(2,jj);
      for(int i=0;i<=h-deltah;i+=deltah){
	for(int j=0;j<=w-deltaw;j+=deltaw){
	  cout << i << "," << j << "," << deltah << "," << deltaw << "," << endl;
	  temp = img.clone();
	  temp = temp(Rect(j,i,deltaw,deltah));
	  radon = radonTransform(temp);
	  lrbp = elbp(radon, 2, 8);
	  lrbp.convertTo(lrbp, CV_8U);
	  /// Compute the histograms:
	  calcHist( &lrbp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
	  hist.convertTo(hist, CV_32F);
	  if(result.empty())
	    result = hist.clone();
	  else
	    vconcat(result.clone(), hist.clone(), result);
	  cout << result.size() << endl;
	}
      }
      photosHist[ii].push_back(result.clone());
      cout << photosHist[ii].size() << endl;
    }
    cout << ii << endl;;
  }
  
  cout << endl <<"calc sketches" << endl;
  
  for(int ii=0; ii<sketches.size(); ii++){
    Mat img = sketches[ii], temp, radon, lrbp, hist;
    int w = img.cols, h=img.rows;
    for(int jj=0; jj<3; jj++){
      Mat result;
      int deltaw = w/pow(2,jj), deltah = h/pow(2,jj);
      for(int i=0;i<=h-deltah;i+=deltah){
	for(int j=0;j<=w-deltaw;j+=deltaw){
	  cout << i << "," << j << "," << deltah << "," << deltaw << "," << endl;
	  temp = img.clone();
	  temp = temp(Rect(j,i,deltaw,deltah));
	  radon = radonTransform(temp);
	  lrbp = elbp(radon, 2, 8);
	  lrbp.convertTo(lrbp, CV_8U);
	  /// Compute the histograms:
	  calcHist( &lrbp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
	  hist.convertTo(hist, CV_32F);
	  if(result.empty())
	    result = hist.clone();
	  else
	    vconcat(result.clone(), hist.clone(), result);
	  cout << result.size() << endl;
	}
      }
      sketchesHist[ii].push_back(result.clone());
      cout << sketchesHist[ii].size() << endl;
    }
    cout << ii << endl;;
  }
  
  vector<int> rank(nSketches);
  
  cerr << endl << "calculating distances" << endl;
  
  for(int i=0; i<nSketches; i++){
    float val = 0;
    val += chiSquareDistance(photosHist[i][0],sketchesHist[i][0])/pow(2,3);
    for(int k=1; k<3; k++)
      val += chiSquareDistance(photosHist[i][k],sketchesHist[i][k])/pow(2,3-k+1);
    cerr << "photo and sketch "<< i << " d1= "<< val << endl;
    int temp = 0;
    for(int j=0; j<nPhotos; j++){
      float localVal = 0;
      localVal += chiSquareDistance(photosHist[j][0],sketchesHist[i][0])/pow(2,3);
      for(int k=1; k<3; k++)
	localVal += chiSquareDistance(photosHist[j][k],sketchesHist[i][k])/pow(2,3-k+1);
      if(localVal<= val && i!=j){
	cerr << "small "<< j << " d1= "<< localVal << endl;
	temp++;
      }
    }
    rank[i] = temp;
    cerr << i << " rank= " << temp << endl;
  }
  
  for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100})
  {
    cerr << "Rank "<< i << ": ";
    cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nSketches << endl;
  }
  
  return 0;
}