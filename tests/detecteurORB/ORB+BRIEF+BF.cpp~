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

const char* transparency_window = "transparence ORB+BRIEF+BF";
const char* matches_window = "ORB+BRIEF+BF";
 

int thresh = 0;
int max_thresh = 100;


  vector<KeyPoint> keypoints1,keypoints2;
  //vector<DMatch>  matches;
  vector<DMatch> matches;
  
  //vector<vector<DMatch> > matchesWithDist;
  vector<DMatch> matchesWithDist;

   /*
    C++: ORB::ORB(
int nfeatures=500, 
float scaleFactor=1.2f, 
int nlevels=8, 
int edgeThreshold=31, 
int firstLevel=0, 
int WTA_K=2, 
int scoreType=ORB::HARRIS_SCORE, 
int patchSize=31)


    Parameters:	

        nfeatures – The maximum number of features to retain.
        scaleFactor – Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        nlevels – The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels).
        edgeThreshold – This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        firstLevel – It should be 0 in the current implementation.
        WTA_K – The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        scoreType – The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.
        patchSize – size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
*/


  ORB detector(200);
  
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

  Mat image1_grey;
  Mat image2_grey;
  
  cvtColor( image1, image1_grey, COLOR_BGR2GRAY );
  cvtColor( image2,image2_grey, COLOR_BGR2GRAY );


  
  detector(image1_grey,Mat(),keypoints1);
  detector(image2_grey,Mat(),keypoints2);

  //detector.detect(image1,keypoints1);
  //detector.detect(image2,keypoints2);

  
  Mat descriptors1,descriptors2;
  descriptor.compute(image1,keypoints1,descriptors1);
  descriptor.compute(image2,keypoints2,descriptors2);

  
  
  // Construction of the matcher
  //BruteForceMatcher< HammingLUT > matcher;
  BruteForceMatcher<Hamming> matcher;// =BruteForceMatcher<Hamming>(10);

  Mat descriptorAuxKp1;
  Mat descriptorAuxKp2;


  vector<int> associateIdx;

  for(int i=0;i<descriptors1.rows;i++){
   //on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le Keypoints[i]
   descriptors1.row(i).copyTo(descriptorAuxKp1);

//ici on va mettre que les valeurs du descripteur des keypoints de l'image 2 que l'on veut comparer aux keypoints de l'image1 en cours de traitement
   descriptorAuxKp2.create(0,0,CV_8UC1);
   

   //associateIdx va servir à faire la transition entre les indices renvoyés par matches et ceux des Keypoints
   associateIdx.erase(associateIdx.begin(),associateIdx.end());

    
    for(int j=0;j<descriptors2.rows;j++){

      float p1x=keypoints1[i].pt.x;
      float p1y=keypoints1[i].pt.y;
      float p2x=keypoints2[j].pt.x;
      float p2y=keypoints2[j].pt.y;

      float distance=sqrt(pow((p1x-p2x),2)+pow((p1y-p2y),2));
      
	//parmis les valeurs dans descriptors2 on ne va garder que ceux dont les keypoints associés sont à une distance définie du keypoints en cours, en l'occurence le ieme ici.
      if(distance<4){
       
        descriptorAuxKp2.push_back(descriptors2.row(j));
        associateIdx.push_back(j);
       
      }
      

    }/*
    cout<<"\nk1 ";
    cout<<"\nangle " << keypoints1[0].angle;
    cout<<"\nclass_id " << keypoints1[0].class_id;
    cout<<"\noctave " << keypoints1[0].octave ;
    cout<<"\npt " << keypoints1[0].pt ;
    cout<<"\nresponse " << keypoints1[0].response;
    cout<<"\nsize " << keypoints1[0].size;
    */


    //ici on ne matche qu'un keypoints de l'image1 avec tous les keypoints gardés de l'image 2
    matcher.match(descriptorAuxKp1,descriptorAuxKp2, matches);
    
    //on remet à la bonne valeur les attributs de matches
    for(int idxMatch=0;idxMatch<matches.size();idxMatch++){
	//on a comparer le keypoints i
      matches[idxMatch].queryIdx=i;
        //avec le keypoints2 j
      matches[idxMatch].trainIdx=associateIdx[matches[idxMatch].trainIdx];
    }
    
	//on concatene les matches trouvés pour les points précedents avec les nouveaux
    matchesWithDist.insert(matchesWithDist.end(),matches.begin(),matches.end());
   

  }



//ici on trie les matchesWithDist par distance des valeurs des descripteurs et non par distance euclidienne
  nth_element(matchesWithDist.begin(),matchesWithDist.begin()+24,matchesWithDist.end());  
    // initial position
 // position of the sorted element
             // end position

Mat imageMatches;
Mat matchesMask;
drawMatches(
  image1,keypoints1, // 1st image and its keypoints
  image2,keypoints2, // 2nd image and its keypoints
  matchesWithDist,            // the matches
  imageMatches,      // the image produced
  Scalar::all(-1),   // color of the lines
  Scalar(255,255,255) //color of the keypoints
  );


  namedWindow( matches_window, CV_WINDOW_AUTOSIZE );
  imshow( matches_window, imageMatches );
  imwrite("resultat.png", imageMatches);
  

  
  /// Create a window and a trackbar
  namedWindow(transparency_window, WINDOW_AUTOSIZE );
  createTrackbar( "Threshold: ", transparency_window, &thresh, max_thresh, interface );
  //imshow(transparency_window,image1 );







  interface( 0, 0 );

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
