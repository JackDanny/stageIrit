#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std; 

/// Global variables

///Matrix of the two images
Mat image1, image2;

///name of the window with a trackbar
string transparency_window;

///name of the window with the two images, their KeyPoints and their matches
string matches_window;

///initial value of the thresh at the begining at the launching of the interface
int thresh = 0;

///value max of the trackbar
int max_thresh = 100;

///value of the maximal matchingDistance we want. MatchingDistance can be consider like 
///the evaluation of similarity between two points.
float threshMatches=50;

///keypoints1 correspond to keypoints detected in the 1st image
///pointsx correspond to the keypoints detected in the 1st image keeped to be matched
///pointsy correspond to the keypoints detected in the 2nd image keeped to be matched
vector < KeyPoint > keypoints1, keypoints2, pointsx, pointsy;


//And we have checked that in matching pointsy  with the 1st image, we  retrieve the same point 
//the 2nd matches founded pointsx2 and pointsy2
//pointsy2 is a subset of pointsy
//pointsx2 is a subset of pointsx
vector < KeyPoint > pointsy2,pointsx2; 



///vector of the matches of one keypoint.
vector < DMatch > matches;

///vector of the matches of keypoints
vector < DMatch > matchesWithDist2;



/// Function header

///For display the window with trackbar
///it's a callback fonction call each time we move the cursor of the trackbar
void interface(int argc, void *);

//function for know the datatype of members in one matrix, e.g CV_8UC1
string type2str(int type);


/**
 * @function main
 */

int main(int argc, char **argv)
{


    Ptr < DescriptorExtractor > descriptor;
    Ptr < DescriptorMatcher > matcher;

    ///checking the number of argument
   if(argc!=6)
   {
        cout << "\nusage incorrect!\n\n";
        cout << "pathimage1 pathimage2 detectorName descriptorName matcherName\n" <<endl;

        cout << "\n detectorName List:";
        cout << "\n MSER";
        cout << "\n FAST";
        cout << "\n STAR";
        cout << "\n SIFT";
        cout << "\n SURF";
        cout << "\n ORB";
        cout << "\n HARRIS";
        cout << "\n GFTT";
        cout << "\n DENSE";
        cout << "\n SIMPLE BLOB\n";
        cout << endl;

        cout << "\n descriptorName List:";
        cout << "\n SIFT";
        cout << "\n SURF";
        cout << "\n BRIEF";
        cout << "\n ORB";
        cout << endl;

        cout << "\n matcherName List:";

        cout << "\n BruteForce";
        cout << "\n BruteForce-L1";
        cout << "\n BruteForce-Hamming";
        cout << "\n FlannBased";
        cout <<"\n";
        cout <<endl;
        

        exit(-1);
    }



    image1 = imread(argv[1], 1);
	///checking pathimage1 exist
    if(! image1.data)
    {
        cout << endl << endl;
        cout <<"error: in"<<endl;
        cout <<argv[0];
        cout << " pathimage1 pathimage2 detectorName descriptorName matcherName\n" <<endl;
	cout << endl << "argument pathimage1 \""<< argv[1] << "\" is not valid";
        cout << endl;
        exit(-1);
    }

    image2 = imread(argv[2], 1);
	///checking pathimage2 exist
     if(! image2.data)
    {
        cout << endl << endl;
        cout <<"error: in"<<endl;
        cout <<argv[0];
        cout << " pathimage1 pathimage2 detectorName descriptorName matcherName\n" <<endl;
	cout << endl << "argument pathimage2 \""<< argv[2] << "\" is not valid";
        cout << endl;
        exit(-1);
    }

    ///selection of the detector in function of argv[3]
    if (strcmp(argv[3], "MSER") == 0) 
    {
	MserFeatureDetector detector;
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }


    else if (strcmp(argv[3], "FAST") == 0) 
    {
	FastFeatureDetector detector(50, true);
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }

    else if (strcmp(argv[3], "STAR") == 0) 
    {
	StarFeatureDetector detector;
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }


    else if (strcmp(argv[3], "SIFT") == 0) 
    {

	SiftFeatureDetector detector;
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);

	///or
	/*
        SIFT detector;
	detector(image1,Mat(),keypoints1);
	*/ 

    }

    else if (strcmp(argv[3], "SURF") == 0) {

	SurfFeatureDetector detector;
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);

        ///or
	/*
	   SURF detector(100);
	   detector(image1,Mat(),keypoints1);
	 */

    }
    else if (strcmp(argv[3], "ORB") == 0) 
    {
	/*ORB detector;
	detector(image1, Mat(), keypoints1);
        detector(image2, Mat(), keypoints2);
	*/
        ///or
	
	   OrbFeatureDetector detector;
	   detector.detect(image1, keypoints1);
	   detector.detect(image2, keypoints2);
    }


    else if (strcmp(argv[3], "HARRIS") == 0) 
    {
	GoodFeaturesToTrackDetector detector(1000, 0.01, 1., 3, true, 0.04);
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);

    }
    else if (strcmp(argv[3], "GFTT") == 0) 
    {
	GoodFeaturesToTrackDetector detector;
	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }

    else if (strcmp(argv[3], "DENSE") == 0) 
    {
	Ptr < FeatureDetector > detector;
	detector = FeatureDetector::create("Dense");
        
        ///other usage:
	//DenseFeatureDetector detector(0.5);
	detector->detect(image1, keypoints1);
        detector->detect(image2, keypoints2);

    } 
    else if (strcmp(argv[3], "SIMPLE BLOB") == 0) {
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 10.0;     // minimum 10 pixels between blobs
	params.filterByArea = true;	       // filter my blobs by area of blob
	params.minArea = 10.0;	               // min 20 pixels squared
	params.maxArea = 500.0;	               // max 500 pixels squared
	SimpleBlobDetector detector(params);

	detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }

    ///checking detector
    else
    {

        cout << "\n detector unknow"<<endl;
        cout << "\n detectors valid list:";
        cout << "\n MSER";
        cout << "\n FAST";
        cout << "\n STAR";
        cout << "\n SIFT";
        cout << "\n SURF";
        cout << "\n ORB";
        cout << "\n HARRIS";
        cout << "\n GFTT";
        cout << "\n DENSE";
        cout << "\n SIMPLE BLOB";
        cout <<"\n";
        exit(-1);
    }


    //descriptor/extractor selection in function of argv[4]



    if (strcmp(argv[4], "BRIEF") == 0) 
    {
	descriptor = new BriefDescriptorExtractor(64);
        //or
        //descriptor = cv::DescriptorExtractor::create("BRIEF");

    } 
    else if (strcmp(argv[4], "ORB") == 0) 
    {
	descriptor = new OrbDescriptorExtractor();
        //or
        //descriptor = cv::DescriptorExtractor::create("ORB");

    }
    else if (strcmp(argv[4], "SIFT") == 0) 
    {
	descriptor = new SiftDescriptorExtractor();
        //or
	//descriptor = cv::DescriptorExtractor::create("SIFT");

    }
    else if (strcmp(argv[4], "SURF") == 0) 
    {
	descriptor = new SurfDescriptorExtractor();
        //or
	//descriptor = cv::DescriptorExtractor::create("SURF");

    }

    ///checking descriptor
    else
    {

        cout << "\n descriptor/extractor unknow"<<endl;
        cout << "\n descriptor/extractor valid list:";
        cout << "\n SIFT";
        cout << "\n SURF";
        cout << "\n BRIEF";
        cout << "\n ORB";
        cout <<"\n";
 
        exit(-1);
    }

     ///selection of matcher in function of argv[5]
     ///Be careful with the type of data of members of the descriptors matrix and the type of
     ///data valid by the matcher. Commons confusion are due to CV_8UC1 and CV_32FC1 
    if (strcmp(argv[5], "BruteForce") == 0 || strcmp(argv[5], "BruteForce-L1") == 0 || strcmp(argv[5], "BruteForce-Hamming") == 0 || strcmp(argv[5], "FlannBased") == 0) 
    {
	matcher = cv::DescriptorMatcher::create(argv[5]);
    }
  
    ///checking matcher
    else{
        cout << "\n matcher unknow"<<endl;
        cout << "\n matcher valid list:";
        cout << "\n BruteForce";
        cout << "\n BruteForce-L1";
        cout << "\n BruteForce-Hamming";
        cout << "\n FlannBased";
        cout <<"\n";
 
        exit(-1);
     

    }


    ///part of title of the windows composed by detector+descriptor+matcher,
    ///for know what is what
    string st1 = argv[3] + (string) " + " + argv[4] + (string) " + " + argv[5];
    string st2 = "transparence " + st1;

     
    transparency_window = st2;
    matches_window = st1;

    
    //Displaying of the two images
    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);

    ///descriptors computation	
    ///the descriptors have one index corresponding to the number of the keypoint (his place in the vector)
    ///and the other index for the value of one component of the vector used by the descriptor
    ///algorithm
    Mat descriptors1; //for 1st image
    Mat descriptors2; //for 2nd image


    cout << endl <<"nb de kp1 before descriptor:"<< keypoints1.size()<<endl;

    (*descriptor).compute(image1, keypoints1, descriptors1);

    cout << endl <<"nb de kp1 after descriptor:"<< keypoints1.size()<<endl;
    
    cout << endl <<"nb de kp2 before descriptor:"<< keypoints2.size()<<endl;

    (*descriptor).compute(image2, keypoints2, descriptors2);

    cout << endl <<"nb de kp2 after descriptor:"<< keypoints2.size()<<endl;

    string ty = type2str(descriptors1.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), descriptors1.cols, descriptors1.rows);
    


    ///descriptor auxiliary to one keypoint of image1. It's a strategy for use the matcher
    Mat descriptorAuxKp1;
   

    //number of points matched. It's need to know who is matched with who
    int currentPoint = 0;

    for (int i = 0; i < keypoints1.size(); i++) {
	
	//We copy the line i of the descriptor,corresponding to values give by the descriptor for the Keypoints[i]
	descriptors1.row(i).copyTo(descriptorAuxKp1);

        ///here we match only one keypoint1 with the best of descriptors2



        matcher->match(descriptorAuxKp1, descriptors2, matches);
 
	///we need to have a no empty vector
	if(matches.size()>0){
       		///we add the KeyPoint in the list
        	pointsx.push_back(keypoints1[i]);
            
		KeyPoint kp2 = keypoints2[matches[0].trainIdx];
        	///we add the KeyPoint in the list
                pointsy.push_back(kp2);
                
                //ready for the next point!
		currentPoint++;
	}




    }				//end of "for i"

   

//let's go for start in the opposite direction. The pixels to match are in pointsy (2nd Image)







//mark for separate the two treatments
/**********************************************************************************************/

//why not reuse our old variables?
currentPoint = 0;

///descriptor for the pointsy
 Mat descriptorsPtsy;

///descriptor for the pointsx
 Mat descriptorsPtsx;

 ///descriptor auxiliary to one keypoint of image2. It's a strategy for use the matcher
 Mat descriptorAuxKp2;

//Warning! we calculate the image2 descriptor here
 (*descriptor).compute(image2, pointsy, descriptorsPtsy);

///cout<<"descriptors ptsx size: "<<descriptorsPtsy.cols<<"x"<<descriptorsPtsy.rows<<endl;
//Warning! we calculate the image1 descriptor here
 (*descriptor).compute(image1, pointsx, descriptorsPtsx);



/*


sort(pointsy.begin(),pointsy.end());
 for (int i = 0; i < pointsy.size(); i++) {
cout<< "point "<<i<<" x="<<pointsy[i].pt.x<<endl;
cout<< "point "<<i<<" y="<<pointsy[i].pt.y<<endl;
}
*/


 for (int i = 0; i < pointsy.size(); i++) {

 	//We copy the line i of the descriptor,corresponding to values give by the descriptor for the pointsy[i].
	    descriptorsPtsy.row(i).copyTo(descriptorAuxKp2);
            

	    matcher->match(descriptorAuxKp2, descriptorsPtsx, matches);
 

	    //we need to have a no empty vector
	    if(matches.size()>0){

               //we found the best keypoint
	       KeyPoint kpfind = pointsx[matches[0].trainIdx];

	       //we compare the keypoint found
               if( abs(kpfind.pt.x - pointsx[i].pt.x)<=1  && abs(kpfind.pt.y - pointsx[i].pt.y)<=1 ){
                  
		
                   //good! in addition to this, we know that for each z < pointsx.size and > 0,
                   ///pointsx2[z] is matched with poitsy2[z]. So
                   matches[0].trainIdx = currentPoint;
	           matches[0].queryIdx = currentPoint;

		   //we keep only enough good matches 
                   if(matches[0].distance<= threshMatches)
                   {
/*
                       cout<<endl<<kpfind.pt.x<<endl;
                   cout<<pointsx[i].pt.x<<endl;

                   cout<<kpfind.pt.y<<endl;
          	   cout<<pointsx[i].pt.y<<endl;
     */              
                      ///we add our points in the lists
                      pointsx2.push_back(kpfind);
                      pointsy2.push_back(pointsy[i]);
                
                      matchesWithDist2.insert(matchesWithDist2.end(), matches.begin(), matches.end());
                      currentPoint++;
                   }
           }

         }


    }				//end of "for i"


/*********************************************************************************************/









   cout << endl<< "nb of matching: " << matchesWithDist2.size() << endl;



   
    ///matches are sorted with the help of the definition of the operator "<". Here it 
    ///means with the distanceMatching value. Indeed, a DMatch has a field distance
    sort(matchesWithDist2.begin(),matchesWithDist2.end());
    // initial position
    // end position

    
    //if we want to keep only some matches
    /* if(matches.size()>500){
       matchesWithDist.erase(matchesWithDist.begin() + 500,matchesWithDist.end());
     }*/


    ///the matrix composed by the two images, the pointsx and pointsy
    ///the matches are colored, there is no keypoints displayed no matched
    ///the image is often hard to interpret due to the great number of element to display
    Mat imageMatches;

    drawMatches(image1, pointsx2,	// 1st image and its keypoints
		image2, pointsy2,	// 2nd image and its keypoints
		matchesWithDist2,	// the matches
		imageMatches,	// the image produced
		Scalar::all(-1),	// color of the lines
		Scalar(255, 255, 255)	//color of the keypoints
	);

    ///we display the result and we create a image png named "result" in the local repository 
    namedWindow(matches_window, CV_WINDOW_AUTOSIZE);
    imshow(matches_window, imageMatches);
    imwrite("result.png", imageMatches);



    /// Create a window and a trackbar
    namedWindow(transparency_window, WINDOW_AUTOSIZE);

    ///each time we move the trackbar cursor, the value of thresh change and
    ///the fonction interface is call back
    createTrackbar("Threshold: ", transparency_window, &thresh, max_thresh, interface);
    
    ///interface is a recursive function allowing to use the trackbar
    interface(0, 0);

    ///We wait the user press a key
    waitKey(0);
    return (0);
}

void interface(int, void *)
{

 
    ///Matrix destination of the image we want to display
    Mat dst;

    ///at the begining we see the image1
    image1.copyTo(dst);

    //With the use of the cursor of the trackbar (value of thresh), we give weight to
    ///pixels of each image with help of linear interpolation
    for (int i = 0; i < image1.rows; i++) {
	for (int j = 0; j < image1.cols; j++) {

	    dst.at < cv::Vec3b > (i, j)[0] = (float) (image2.at < cv::Vec3b > (i, j)[0]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[0]) * (float) ((100. - thresh) / 100.);
	    dst.at < cv::Vec3b > (i, j)[1] = (float) (image2.at < cv::Vec3b > (i, j)[1]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[1]) * (float) ((100. - thresh) / 100.);
	    dst.at < cv::Vec3b > (i, j)[2] = (float) (image2.at < cv::Vec3b > (i, j)[2]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[2]) * (float) ((100. - thresh) / 100.);


	}
    }
    ///coordonates of KeyPoint1 (image 1)
    float kp1x;
    float kp1y;

    ///coordonates of KeyPoint2 (image 2)
    float kp2x;
    float kp2y;

    ///coordonates of KeyPoint (image dst)
    float kptx;
    float kpty;


    for (int i = 0; i < matchesWithDist2.size(); i++) {

        ///we calculate the coordonates of the two points to matched
	kp1x = pointsx2[matchesWithDist2[i].queryIdx].pt.x;
	kp1y = pointsx2[matchesWithDist2[i].queryIdx].pt.y;

	kp2x = pointsy2[matchesWithDist2[i].trainIdx].pt.x;
	kp2y = pointsy2[matchesWithDist2[i].trainIdx].pt.y;

         //and we calculate the position of the point to display in the dst image
        //by linear interpolation with the help of thresh
	kptx = kp1x * (100. - thresh) / 100. + kp2x * (thresh / 100.);
	kpty = kp1y * (100. - thresh) / 100. + kp2y * (thresh / 100.);

	Point ptkp1 = Point(kptx, kpty);

         ///we use RGB with one Byte for each color. So the number of color is...
	int nbColor = 256 * 256 * 256;

        ///we have matchesWithDist.size() points to display. So:
	int ColStep = nbColor / matchesWithDist2.size();

	///We can also divided by matches.size()-1 because point 0 to matches.size()-1, 
        ///but if matches.size()==1, it is not cool

        ///colActu is a value used for calculate the color of the point corresponding to
        ///the matches number i
        int colActu = ColStep * i;

        ///value of blue component
        int blue = colActu % 256;

        int qb = colActu / 256;

	//guess
        int green = qb % 256;
	
	
        int red = qb / 256;

	
	//so we draw the point
        circle(dst, ptkp1, 5, Scalar(red, green, blue), 2, 8, 0);
    }
    //and we display the beautiful colored window
    namedWindow(transparency_window, WINDOW_AUTOSIZE);
    imshow(transparency_window, dst);
    //we decide to write the actual image dst for use illustration
    imwrite("trans.png", dst);
}


string type2str(int type)
{
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
	r = "8U";
	break;
    case CV_8S:
	r = "8S";
	break;
    case CV_16U:
	r = "16U";
	break;
    case CV_16S:
	r = "16S";
	break;
    case CV_32S:
	r = "32S";
	break;
    case CV_32F:
	r = "32F";
	break;
    case CV_64F:
	r = "64F";
	break;
    default:
	r = "User";
	break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}


