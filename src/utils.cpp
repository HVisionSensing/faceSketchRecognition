#include "utils.hpp"

void loadImages(string src, vector<Mat> &dest, float proportion){
  directory_iterator end;
  
  path dir(src);
  
  string filename;
  int n=0;
  int num;
  Mat temp;
  
  for (directory_iterator pos(dir); pos != end; ++pos){
    if(is_regular_file(*pos)){
      n++;
    }
  }
  
  dest.resize(n);
  
  for (directory_iterator pos(dir); pos != end; ++pos){
    if(is_regular_file(*pos)){
      filename = pos->path().filename().string();
      num = atoi((filename.substr(0,filename.find("."))).c_str());
      temp = imread(pos->path().c_str(), 0);
      resize(temp, temp, Size(temp.size().width*proportion,
			      temp.size().height*proportion));
      dest[num-1] = Mat(temp(Rect(temp.cols/2-100,
				  temp.rows/2-125, 200, 250)));
    }
  }
}


void createFolds(vector<Mat>& input, vector<vector<Mat> >&output, int num){
  vector<Mat>::iterator it;
  int size = (input.size()/num)*num;
  
  for(int i=0; i<num; i++){
    output.push_back(vector<Mat>());
  }
  
  for(int i = 0; i < size; i++) {
    output[i%num].push_back(input[i]);
  }
}

void patcher(Mat img, int size, int delta, vector<vector<Mat> > &result){
  int w = img.cols, h=img.rows;
  
  for(int i=0;i<=h-size;i+=(size-delta)){
    vector<Mat> col;
    for(int j=0;j<=w-size;j+=(size-delta)){
      col.push_back(img(Rect(j,i,size,size)));
    }
    result.push_back(col);
  }
  
}

float chiSquareDistance(Mat a, Mat b){
  float result = 0;
  for (int i = 0; i < a.rows; i++){
    float temp = pow((a.at<float>(i) - b.at<float>(i)),2)/(a.at<float>(i) + b.at<float>(i));
    result += temp!=temp? 0 : temp;
  }
  return result;
}