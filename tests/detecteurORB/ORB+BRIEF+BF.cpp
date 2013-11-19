#include <opencv2/opencv.hpp>

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/core/mat.hpp"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat image1, image2;
int rows;
int cols;

const char* transparency_window = "transparence";
const char *matches_window = "Harris+BRIEF+BF";

int thresh = 0;
int max_thresh = 100;


  vector<KeyPoint> keypoints1,keypoints2;
  //vector<DMatch>  matches;
  vector<DMatch> matches;
  
  //vector<vector<DMatch> > matchesWithDist;
  vector<DMatch> matchesWithDist;


  /*
   *cv::StarFeatureDetector::StarFeatureDetector (
	int maxSize,
	int responseThreshold = 30,
	int lineThresholdProjected = 10,
	int lineThresholdBinarized = 8,
	int suppressNonmaxSize = 5	 
   ) 	
   */
    //OrbFeatureDetector detector;

    ORB detector;

    //Ptr<FeatureDetector> detector = FeatureDetector::create("ORB" ); 
    


    BriefDescriptorExtractor descriptor(64);



/// Function header
void interface( int argc, void* );

/**
 * @function main
 */
int main( int, char** argv )
{
  


  image1 = imread( argv[1], 1 );
  image2 = imread( argv[2], 1 );
  rows=image1.rows;
  cols=image1.cols;

  
  namedWindow( "image1", WINDOW_AUTOSIZE );
  imshow( "image1",image1 );
  namedWindow( "image2", WINDOW_AUTOSIZE );
  imshow( "image2",image2 );

  Mat descriptors1,descriptors2;
  
    
  Mat image1_grey;
  Mat image2_grey;
/*
 cvtColor(image1, image1_grey, COLOR_BGR2GRAY);
 cvtColor(image2, image2_grey, COLOR_BGR2GRAY);
*/
  

  detector(image1,image1,keypoints1,descriptors1);
  detector(image2,image2,keypoints2,descriptors2);



  waitKey(0);
  return(0);
}

/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */
void interface( int, void* )
{

  Mat dst;
  image1.copyTo(dst);

  ///on adapte l'importance des pixels de chaque image selon la valeur du trackbar
  for(int i=0;i<rows;i++){
     for(int j=0;j<cols;j++){

       dst.at<cv::Vec3b>(i,j)[0]= (float)(image2.at<cv::Vec3b>(i,j)[0])*(float)(thresh/100.) +(float)( image1.at<cv::Vec3b>(i,j)[0])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[1]=(float)(image2.at<cv::Vec3b>(i,j)[1])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[1])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[2]=(float)(image2.at<cv::Vec3b>(i,j)[2])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[2])*(float)((100.-thresh)/100.)  ;

  
     }
  }

  float kp1x;
  float kp1y;
  float kp2x;
  float kp2y;

  float kptx;
  float kpty;
  
  vector<KeyPoint> keypointsKeep1,keypointsKeep2;
  

  for(int i=0;i<matchesWithDist.size();i++){
    
    
    kp1x=keypoints1[matchesWithDist[i].queryIdx].pt.x;
    kp1y=keypoints1[matchesWithDist[i].queryIdx].pt.y;
    kp2x=keypoints2[matchesWithDist[i].trainIdx].pt.x;
    kp2y=keypoints2[matchesWithDist[i].trainIdx].pt.y;
    
    kptx=kp1x*(100.-thresh)/100. + kp2x*(thresh/100.);
    kpty=kp1y*(100.-thresh)/100. + kp2y*(thresh/100.);

    Point ptkp1=Point(kptx,kpty);

    int nbColor=256*256*256;

    int pascoul=nbColor/matchesWithDist.size();
    int coulActu=pascoul*i;
    
    int bleu=coulActu%256;
    
    int qb=coulActu/256;
    int vert=qb%256;

    int rouge=qb/256;
    
    

    circle( dst, ptkp1, 5,  Scalar(rouge,vert,bleu), 2, 8, 0 );
  }
  
  namedWindow( transparency_window, WINDOW_AUTOSIZE );
  imshow(transparency_window, dst );
}
