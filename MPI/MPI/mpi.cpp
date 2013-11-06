#include <stdio.h>
#include <opencv2/opencv.hpp>

/*
g++ mpi.cpp `pkg-config --cflags --libs opencv`

#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/legacy/legacy.hpp"
*/
using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
  Mat image1, image2;
  image1 = imread( argv[1], 1 );
  image2 = imread( argv[2], 1 );

  if( argc != 3 || !image1.data || !image2.data)
    {
      printf( "No image data \n" );
      return 1;
    }

// vector of keypoints
      std::vector<cv::KeyPoint> keypoints1, keypoints2;
      // Construct the SURF feature detector object
      cv::SurfFeatureDetector surf(
                2500.); // threshold
      // Detect the SURF features
      surf.detect(image1,keypoints1);
      surf.detect(image2,keypoints2);

//cv::drawKeypoints(image1,
//   keypoints1,
//   image1,
//   cv::Scalar(255,255,255), // keypoint color
//   //cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag
//   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
//cv::drawKeypoints(image2,
//   keypoints2,
//   image2,
//   cv::Scalar(255,255,255), // keypoint color
//   //cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag
//   cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag


  namedWindow( "Image 1", CV_WINDOW_AUTOSIZE );
  imshow( "Image 1", image1 );
  namedWindow( "Image 2", CV_WINDOW_AUTOSIZE );
  imshow( "Image 2", image2 );
  
  // Construction of the SURF descriptor extractor
      cv::SurfDescriptorExtractor surfDesc;
      // Extraction of the SURF descriptors
      cv::Mat descriptors1, descriptors2;
      surfDesc.compute(image1,keypoints1,descriptors1);
      surfDesc.compute(image2,keypoints2,descriptors2);
      
      // Construction of the matcher
cv::BruteForceMatcher<cv::L2<float> > matcher;
// Match the two image descriptors
std::vector<cv::DMatch> matches;
matcher.match(descriptors1,descriptors2, matches);

 std::nth_element(matches.begin(),    // initial position
          matches.begin()+24, // position of the sorted element
          matches.end());     // end position
      // remove all elements after the 25th
     // matches.erase(matches.begin()+50, matches.end());
cout << '\n' << "nombre de correspondances:" << matches.size() << '\n';      
cv::Mat imageMatches;
cv::drawMatches(
  image1,keypoints1, // 1st image and its keypoints
  image2,keypoints2, // 2nd image and its keypoints
  matches,            // the matches
  imageMatches,      // the image produced
  Scalar::all(-1),   // color of the lines
  Scalar(255,255,255) //color of the keypoints
); // color of the lines

 namedWindow( "Matches SURF", CV_WINDOW_AUTOSIZE );
  imshow( "Matches SURF", imageMatches );
    cv::imwrite("resultat.png", imageMatches);   
  

  waitKey(0);

  return 0;
}
