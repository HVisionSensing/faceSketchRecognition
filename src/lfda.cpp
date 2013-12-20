#include "lfda.hpp"

LFDA::LFDA(vector<Mat> &trainingPhotos, vector<Mat> &trainingSketches, int size, int overlap)
{
  this->trainingPhotos=trainingPhotos;
  this->trainingSketches=trainingSketches;
  this->size = size;
  this->overlap = overlap;
}

LFDA::~LFDA()
{
  
}

void LFDA::compute()
{
  for(int i=0; i < this->trainingSketches.size(); i++){
    vector<Mat> phi = this->extractDescriptors(this->trainingSketches[i],this->size,this->overlap);
    for(int j=0; j<phi.size(); j++){
      if(i==0){
	this->Xsk.push_back(phi[j]);
	this->Xk.push_back(phi[j]);
      }
      else{
	hconcat(this->Xsk[j], phi[j], this->Xsk[j]);
	hconcat(this->Xk[j], phi[j], this->Xk[j]);
      }
    }
    _classes.push_back(i);
  }
  
  for(int i=0; i < this->trainingPhotos.size(); i++){
    vector<Mat> phi = this->extractDescriptors(this->trainingPhotos[i],this->size,this->overlap);
    for(int j=0; j<phi.size(); j++){
      if(i==0)
	this->Xpk.push_back(phi[j]);
      else
	hconcat(this->Xpk[j], phi[j], this->Xpk[j]);
      
      hconcat(this->Xk[j], phi[j], this->Xk[j]);
    }
    _classes.push_back(i);
  }
  
  
  for(int i=0; i < Xk.size(); i++){
   
    Mat Xk_mean=Xk[i].col(0), Xpk_mean=Xpk[i].col(0), Xsk_mean=Xsk[i].col(0);
       
    int ncols = Xk[i].cols; 
    for(int j=0; j<ncols; j++)
	Xk_mean+=Xk[i].col(j);
    
    Xk_mean = Xk_mean*(1.0/ncols);
    
    XkVectorMean.push_back(Xk_mean.clone());
    
    for(int j=0; j<ncols; j++)
      Xk[i].col(j)-=Xk_mean;

    PCA pca(Xk[i],Mat(),CV_PCA_DATA_AS_COL,100);
    Mat Wk = pca.eigenvectors;
    
    Mat dataPCA = Wk*Xk[i];
    dataPCA = dataPCA.t();
    
    LDA lda;
    lda.compute(dataPCA, _classes);
  
    Mat Lk = lda.eigenvectors();
    Lk.convertTo(Lk, CV_32F);
  
    Mat omega = Lk.t()*Wk;
    
    omegaK.push_back(omega.t());
  }
}

Mat LFDA::project(Mat image)
{
  vector<Mat> phi = extractDescriptors(image,this->size,this->overlap);
  Mat temp = omegaK[0].t()*(phi[0] - XkVectorMean[0]);
  normalize(temp,temp,1);
  Mat result = temp.clone();
  for(int j=1; j<phi.size(); j++){
    temp = omegaK[j].t()*(phi[j]-XkVectorMean[j]);
    normalize(temp,temp,1);
    vconcat(result, temp.clone(), result);
  }
  
  //normalize(result, result, 1);
  
  return result;
}

vector<Mat> LFDA::extractDescriptors(Mat img, int size, int delta){
  int w = img.cols, h=img.rows;
  vector<Mat> result;
  
  for(int i=0;i<=w-size;i+=(size-delta)){
    Mat aux, temp;
    for(int j=0;j<=h-size;j+=(size-delta)){
      Mat a, b;
      calcSIFTDescriptors(img(Rect(i,j,size,size)),a);
      normalize(a,a,1);
      calcLBPHistogram(img(Rect(i,j,size,size)),b);
      normalize(b,b,1);
      hconcat(a,b,temp);
      if(aux.empty())
	aux = temp.clone();
      else
	hconcat(aux, temp, aux);
    }
    result.push_back(aux.t());
  }
  
  return result;
}
