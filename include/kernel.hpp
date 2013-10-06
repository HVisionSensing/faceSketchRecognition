/*
 *    <one line to give the program's name and a brief idea of what it does.>
 *    Copyright (C) 2013  <copyright holder> <email>
 * 
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 * 
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 * 
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
#include "descriptors.hpp"
#include "filters.hpp"

using namespace std;
using namespace cv;

class Kernel
{
private:
  vector<Mat> trainingPhotos,	trainingSketches, trainingPhotosDescriptors, trainingSketchesDescriptors;
  int patches;
  string filter, descriptor;
  Mat Kp, Kg, R, T2, mean;
  Mat projectGalleryIntern(Mat image);
  Mat projectProbeIntern(Mat image);
public:
  PCA pca;
  LDA lda;
  Kernel(vector<Mat>& trainingPhotos,vector<Mat>& trainingSketches, int patches, string filter, string descriptor);
  virtual ~Kernel();
  void compute();
  Mat projectGallery(Mat image);
  Mat projectProbe(Mat image);
  Mat extractDescriptors(Mat img);
  float calcKernel(Mat x, Mat y);
};

#endif // KERNEL_HPP
