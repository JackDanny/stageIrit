#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/core/mat.hpp"

using namespace cv;
using namespace std;




const char* transparency_window = "transparence";

int thresh = 0;
int max_thresh = 100;
int cols;
int rows;

// Match the two image descriptors
vector<DMatch> matches;

/// vector of keypoints 
vector<KeyPoint> keypoints1,keypoints2;


/// Function header
void interface( int argc, void* );
Mat image1,image2;

int main(int argc,char** argv){




const char* source_window = "Source image";

 /// Load images
 image1 = imread( argv[1], 1 );
 image2 = imread( argv[2], 1 );



  if( argc != 3 || !image1.data || !image2.data)
    {
      printf( "No image data \n" );
      return 1;
    }


    int cols=image1.cols;
    int rows=image1.rows;
   //   cout<<"\ntaille de la matrice:" <<image1.size();
  //  cout<<"\ntype de la matrice: \n" << image1.type();
  //  cout<<"\nflags" << image1.flags;
  //  cout<<"\ndims" << image1.dims;
    cout<<"\nrows" << image1.rows;
    cout<<"\ncols" << image1.cols;
    
  //  cout<<"\nnombre de chanels: " << image1.channels();

  //  cout<< "\npoints 1 1 " << (int)image1.at<cv::Vec3b>(0,1)[1];
    
    /*
    for(int i=0;i<cols;i++){
	for(int j=0;j<rows;j++){
		image1.at<cv::Vec3b>(i,j)[0]=0;
		image1.at<cv::Vec3b>(i,j)[1]=0;
		image1.at<cv::Vec3b>(i,j)[2]=0;
	}
    }
    */

 // cout<<"\nimage1" <<  image1; 

 


///Construct the SURF feature detector object
  SiftFeatureDetector sift;

  sift.detect(image1,keypoints1);
  sift.detect(image2,keypoints2);

  namedWindow( "Image1", WINDOW_AUTOSIZE );
  imshow( "Image1", image1 );
  namedWindow( "Image2", WINDOW_AUTOSIZE );
  imshow( "Image2", image2 );
  //afficher les coordonées des points des keypoints
	/*for(int i=0;i<keypoints1.size();i++){
        cout<<"\n\nkeypoints number" << i <<"\n";
	cout<<"\nkeypoints1" <<  keypoints1[i].pt; 
  	cout<<"\nkeypoints1x " <<  keypoints1[i].pt.x; 
	cout<<"\nkeypoints1y " <<  keypoints1[i].pt.y; 
         
	}*/


  /*Mat imcopy;
  image1.copyTo(imcopy);
  for(int i=0;i<keypoints1.size();i++){
     imcopy.at<cv::Vec3b>(keypoints1[i].pt.y,keypoints1[i].pt.x)[0]=0;
     imcopy.at<cv::Vec3b>(keypoints1[i].pt.y,keypoints1[i].pt.x)[1]=0;
     imcopy.at<cv::Vec3b>(keypoints1[i].pt.y,keypoints1[i].pt.x)[2]=255;
  }
  namedWindow( "Image copy", CV_WINDOW_AUTOSIZE );
  imshow( "Image copy",  imcopy );
  */

 
  cout << "\ntaille du vecteur de keypoints: " << keypoints1.size(); 

  
  SiftDescriptorExtractor siftDesc;
  
  Mat descriptors1,descriptors2;
  siftDesc.compute(image1,keypoints1,descriptors1);
  siftDesc.compute(image2,keypoints2,descriptors2);
  
   // Construction of the matcher
BruteForceMatcher<L2<float> > matcher;

matcher.match(descriptors1,descriptors2, matches);

nth_element(matches.begin(),    // initial position
          matches.begin()+24, // position of the sorted element
          matches.end());     // end position
      // remove all elements after the 25th
	//display the element attributs
	//cout<< "\nmatches  " <<  matches;
	
	//afficher les matches


	for(int i=0;i<matches.size();i++){
//affichage des attributs
/*		cout<< "\n\npoint num " <<  i;		
		cout<< "\nimgIdx  " <<  matches[i].imgIdx ;	
		cout<< "\nqueryIdx   " <<  matches[i].queryIdx;
		cout<< "\ntrainIdx   " <<  matches[i].trainIdx;
		cout<< "\ndistance   " <<  matches[i].distance;
*/
                
/*
		while(matches[i].distance >100  && i<matches.size()){
			cout << "\ni= " << i;
			matches.erase(matches.begin()+i, matches.begin()+i+1);
		}
           */     
                
	}
        

for(int i=0;i<matches.size();i++){
cout<< "\nOn relie le point de coordonee x1= " << keypoints1[matches[i].queryIdx].pt.x;
		cout<< "\ny1= " << keypoints1[matches[i].queryIdx].pt.y;

		cout<< "\nAvec le point de coordonne x2= " << keypoints2[matches[i].trainIdx].pt.x;
 		cout<< "\ny2= " << keypoints2[matches[i].trainIdx].pt.y;

}
	



      cout << '\n' << "nombre de correspondances:" << matches.size() << '\n';  
/// Create a window and a trackbar
  namedWindow(transparency_window, WINDOW_AUTOSIZE );
  createTrackbar( "Threshold: ", transparency_window, &thresh, max_thresh, interface );	
  interface( 0, 0 );


      

      //matches.erase(matches.begin(), matches.end());
      //keypoints1.erase(keypoints1.begin(), keypoints1.end());
      //keypoints2.erase(keypoints2.begin(), keypoints2.end());
      //matches.erase(matches.begin(), matches.begin()+1600);



  waitKey(0);
  
  return 0;

}




void interface( int, void* )
{

  Mat dst;
  image1.copyTo(dst);

  ///on adapte l'importance des pixels de chaque image selon la valeur du trackbar
  /*for(int i=0;i<rows;i++){
     for(int j=0;j<cols;j++){

       dst.at<cv::Vec3b>(i,j)[0]= (float)(image2.at<cv::Vec3b>(i,j)[0])*(float)(thresh/100.) +(float)( image1.at<cv::Vec3b>(i,j)[0])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[1]=(float)(image2.at<cv::Vec3b>(i,j)[1])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[1])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[2]=(float)(image2.at<cv::Vec3b>(i,j)[2])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[2])*(float)((100.-thresh)/100.)  ;

  
     }
  }*/
 cout<<"\nhouhou!!!";
 for(int i=0;i<rows;i++){
     for(int j=0;j<cols;j++){

       dst.at<cv::Vec3b>(i,j)[0]= (float)(image2.at<cv::Vec3b>(i,j)[0])*(float)(thresh/100.) +(float)( image1.at<cv::Vec3b>(i,j)[0])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[1]=(float)(image2.at<cv::Vec3b>(i,j)[1])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[1])*(float)((100.-thresh)/100.)  ;
       dst.at<cv::Vec3b>(i,j)[2]=(float)(image2.at<cv::Vec3b>(i,j)[2])*(float)(thresh/100.) + (float)(image1.at<cv::Vec3b>(i,j)[2])*(float)((100.-thresh)/100.)  ;

  
     }
  }
  


  namedWindow(transparency_window, WINDOW_AUTOSIZE );
  imshow( transparency_window, dst );




  //line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
  /*float kp1x;
  float kp1y;
  float kp2x;
  float kp2y;*/
  
  /*
  for(int i=0;i<matches.size();i++){
    kp1x=keypoints1[matches[i].queryIdx].pt.x;
    kp1y=keypoints1[matches[i].queryIdx].pt.y;
    kp2x=keypoints2[matches[i].trainIdx].pt.x;
    kp2y=keypoints2[matches[i].trainIdx].pt.y;
    Point pt1=Point(kp1x,kp1y);
    Point pt2=Point(kp2x,kp2y);

    line(dst, pt1, pt2, Scalar(255,255,255));

  }
  */
 
  




}
