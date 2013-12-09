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

int thresh = 0;
int max_thresh = 100;

/// variables for sift

  vector<KeyPoint> keypoints1,keypoints2;
  //vector<DMatch>  matches;
  vector<DMatch> matches;
  
  //vector<vector<DMatch> > matchesWithDist;
  vector<DMatch> matchesWithDist;

///Construct the SIFT feature detector object
  //nrmlt les valeurs par defauts
  //SiftFeatureDetector sift(0.04/3/2.0,10,4,3,0,-1);
  //SiftFeatureDetector sift; //je crois que le descripteur s'adapte dans ce cas
  SiftFeatureDetector sift(0.10,10);
 



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

  

  sift.detect(image1,keypoints1);
  sift.detect(image2,keypoints2);

  BriefDescriptorExtractor briefDesc(64);
  
  Mat descriptors1,descriptors2;
  briefDesc.compute(image1,keypoints1,descriptors1);
  briefDesc.compute(image2,keypoints2,descriptors2);

  
  
  // Construction of the matcher
  //BruteForceMatcher< HammingLUT > matcher;
  BruteForceMatcher<Hamming> matcher;// =BruteForceMatcher<Hamming>(10);

//cout << "\n type= " << descriptors1.type();

/*
  vector<KeyPoint> keypointsAux;
  vector<KeyPoint> keypointsAssoc;
  for(int kp=0;kp<keypoints1.size();kp++){
     keypointsAux.push_back(keypoints[kp]);

    float kpx=keypoints1[kp].pt.x;
    float kpy=keypoints1[kp].pt.y;


    for(int kpa=0;kpa < keypoints2.size();kpa++){
          float kpax=keypoints2[kpa].pt.x;
          float kpay=keypoints2[kpa].pt.y;
	  
          float distance= pow((kpax-kpx),2)+pow((kpx-kpy),2);

        if(distance<10){
		keypointsAssoc.push_back(keypoints2[kpa]);
		
	}

    }
  

  */

  Mat descriptorAuxKp1;

  Mat descriptorAuxKp2;


 //deallocate pop_back(size) 
  vector<int> associateIdx;

  for(int i=0;i<descriptors1.rows;i++){
   
   //descriptorAuxKp1.zeros(300,64,CV_8UC1);
   //descriptors1.copyTo(descriptorAuxKp1);
   //descriptorAuxKp1.create(descriptors1.rows,descriptors1.cols,CV_8UC1);
   descriptors1.row(i).copyTo(descriptorAuxKp1);

  /*
  for(int index1=0;index1<descriptorAuxKp1.rows;index1++){
 
    for(int index2=0;index2<descriptorAuxKp1.cols;index2++){

      descriptorAuxKp1.at<unsigned char>(index1,index2)=0;
    }
 
  }*/
    //cout << descriptorAuxKp1;

   
  
  //descriptorAuxKp1.row(i)= descriptorAuxKp1.row(i)*2;

 
   //descriptorAuxKp1.at<unsigned char>(0,0)=descriptors1.at<unsigned char>(0,0);
   /*for(int i2=0;i2<descriptors1.cols;i2++){
     descriptorAuxKp1.at<unsigned char>(i,i2)=descriptors1.at<unsigned char>(i,i2);
   }*/
   

   //descriptorAuxKp1.row(i)=descriptors1.row(i);

   descriptorAuxKp2.create(0,0,CV_8UC1);
   associateIdx.erase(associateIdx.begin(),associateIdx.end());
    int l=0;
    for(int j=0;j<descriptors2.rows;j++){

      float p1x=keypoints1[i].pt.x;
      float p1y=keypoints1[i].pt.y;
      float p2x=keypoints2[j].pt.x;
      float p2y=keypoints2[j].pt.y;
      
      



      float distance=pow((p1x-p2x),2)+pow((p1y-p2y),2);
      
     

    

      
      if(distance<1000){
        /*cout << "\np1x p2x p1y p2y " << p1x << " " << p2x << " " << p1y << " " << p2y;
        cout << "\ndistance "<<distance;
*/
        cout <<"\non passe la";
        descriptorAuxKp2.push_back(descriptors2.row(j));
        associateIdx.push_back(j);
        cout << associateIdx[l];
        l++;
      }
      
      
      //descriptorAuxKp2.push_back(descriptors2.row(j));
     //cout<<"\ni j " <<i <<" "<<j; 

    }
    //cout << associateIdx;

   // cout << "\n" <<descriptorAuxKp2<<endl;
    
    
    matcher.match(descriptorAuxKp1,descriptorAuxKp2, matches);
    
    //exit(0);

    for(int idxMatch=0;idxMatch<matches.size();idxMatch++){

    matches[idxMatch].queryIdx=i;
    matches[idxMatch].trainIdx=associateIdx[matches[idxMatch].trainIdx];
    
    int kp1x = keypoints1[matches[idxMatch].queryIdx].pt.x;
    int kp1y = keypoints1[matches[idxMatch].queryIdx].pt.y;
    int kp2x = keypoints2[matches[idxMatch].trainIdx].pt.x;
    int kp2y = keypoints2[matches[idxMatch].trainIdx].pt.y;
  // cout << "\np1x p2x p1y p2y " << kp1x << " " << kp2x << " " << kp1y << " " << kp2y;

 
    

    }

    
  
    
    matchesWithDist.insert(matchesWithDist.end(),matches.begin(),matches.end());
    
    cout<<"\ntemps " <<descriptors1.rows-i;


    for(int j2=0;j2< matches.size();j2++){

       int kp1x = keypoints1[matches[j2].queryIdx].pt.x;
       int kp1y = keypoints1[matches[j2].queryIdx].pt.y;
       int kp2x = keypoints2[matches[j2].queryIdx].pt.x;
       int kp2y = keypoints2[matches[j2].queryIdx].pt.y;
 
       //cout << "\np1x p2x p1y p2y " << kp1x << " " << kp2x << " " << kp1y << " " << kp2y;
   


    }


  }

  
  
   /*
   
   for(int j=0;j<descriptors1.cols;j++){
     //descriptorAux.push
     
     cout << "\n point " << i << " " << j << " " << (int)descriptors1.at<unsigned char>(i,j);

    // cout << "\n point " << keypoints1[i].pt.x << " " << keypoints1[i].pt.y << " " << "descripteur " << i <<" "<<j<<" "<< descriptors1.at<long long>(i,j);



     
     
   }*/


  

//}



  //cout<< "\n descriptors1 " << descriptors1 <<endl;
  //cout<< "\ntaille descriptors1 " << descriptors1.ty <<endl;
  //cout<< "\ntaille keypoints1 " << keypoints1.size() <<endl;
  //cout<< "\n descriptors2 " << descriptors2 <<endl;

  // Match the two image descriptors

  //matcher.radiusMatch(descriptors1,descriptors2, matches,10);

/*
  matcher.match(descriptors1,descriptors2, matches);

nth_element(matches.begin(),    // initial position
          matches.begin()+24, // position of the sorted element
          matches.end());     // end position

*/
cout <<"\n matches size= "<<matches.size();

/*

for(int idxMatch=0;idxMatch<matches.size();idxMatch++){
   if(matches[idxMatch].size()!=0){
   cout <<"\n taille de "<< idxMatch <<" "<< matches[idxMatch].size();
   
   for(int i=0;i<matches[idxMatch].size();i++){
     
     cout<<"\n valeur queryIdx "<< matches[idxMatch][i].queryIdx;
     cout<<"\n valeur trainIdx "<< matches[idxMatch][i].trainIdx;
     cout<<"\n valeur kp1x "<< keypoints1[matches[idxMatch][i].queryIdx].pt.x;
     cout<<"\n valeur kp1y "<< keypoints1[matches[idxMatch][i].queryIdx].pt.y;
     cout<<"\n valeur kp2x "<< keypoints2[matches[idxMatch][i].trainIdx].pt.x;
     cout<<"\n valeur kp2y "<< keypoints2[matches[idxMatch][i].trainIdx].pt.y;

   }
   }
   
}
*/


//for(int idxMatch=0;idxMatch<matche

/*
Mat imageMatches;
Mat matchesMask;
drawMatches(
  image1,keypoints1, // 1st image and its keypoints
  image2,keypoints2, // 2nd image and its keypoints
  matches,            // the matches
  imageMatches,      // the image produced
  Scalar::all(-1),   // color of the lines
  Scalar(255,255,255) //color of the keypoints
  );
*/







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








  namedWindow( "Matches BRIEF", CV_WINDOW_AUTOSIZE );
  imshow( "Matches BRIEF", imageMatches );
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
    
    /*
    Point pt1=Point(kp1x,kp1y);
    Point pt2=Point(kp2x,kp2y);

    float distance=pow((kp1x-kp2x),2)+pow((kp1y-kp2y),2);
    */
    /*pour selectioner ceux dont la distance est < 10 pixels 

    while(distance>100 && i<matches.size()){
	matches.erase(matches.begin()+i);
        kp1x=keypoints1[matches[i].queryIdx].pt.x;
        kp1y=keypoints1[matches[i].queryIdx].pt.y;
        kp2x=keypoints2[matches[i].trainIdx].pt.x;
        kp2y=keypoints2[matches[i].trainIdx].pt.y;
        Point pt1=Point(kp1x,kp1y);
        Point pt2=Point(kp2x,kp2y);
        distance=pow((kp1x-kp2x),2)+pow((kp1y-kp2y),2);


    }

    */


    
   /* if(distance <100 && i<matches.size()){
    keypointsKeep1.push_back(keypoints1[matches[i].queryIdx]);
    keypointsKeep2.push_back(keypoints2[matches[i].queryIdx]);
    
    cout<<"\npoint number "<< i;
    cout<<"\nx1 "<<kp1x;
    cout<<"\ny1 "<<kp1y;
    cout<<"\nx2 "<<kp2x;
    cout<<"\ny2 "<<kp2y;
    
    cout<<"\ndistance "<<distance;

    */
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

    //}
 //   line(dst, ptkp1, pt2, Scalar(rouge,vert,bleu),2,8,0);



    //circle( dst, pt1, 5,  Scalar(rouge,vert,bleu), 2, 8, 0 );
    //circle( dst, pt2, 5,  Scalar(rouge,vert,bleu), 2, 8, 0 );
    //circle( dst, pt1, 5,  Scalar(0), 2, 8, 0 );
   


    
    //void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
    //line(dst, pt1, pt2, Scalar::all(-1),1,8,0);

   }

//void drawKeypoints(const Mat& image, const vector<KeyPoint>& keypoints, Mat& outImage, const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT )
   
  // drawKeypoints(dst, keypointsKeep1, dst, Scalar::all(-1));


  
 
  
  namedWindow( transparency_window, WINDOW_AUTOSIZE );
  imshow(transparency_window, dst );
}
