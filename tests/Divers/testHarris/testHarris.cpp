#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main( int argc, char** argv )
{
  Mat image1, image2;
  image1 = imread( argv[1], 1 );

  if( argc != 2 || !image1.data)
    {
      printf( "No image data \n" );
      return 1;
    }

   int blockSize = 2;
  int apertureSize = 3;
  double k = 0.04;

  cornerHarris(image1, image2, blockSize, apertureSize, k, int borderType=BORDER_DEFAULT );


