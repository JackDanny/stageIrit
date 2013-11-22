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

const char *transparency_window = "transparence";
const char *matches_window = "MSER+BRIEF+BF";

int thresh = 0;
int max_thresh = 100;


vector < KeyPoint > keypoints1, keypoints2, pointsx, pointsy;
  //vector<DMatch>  matches;
vector < DMatch > matches;

  //vector<vector<DMatch> > matchesWithDist;
vector < DMatch > matchesWithDist;


  /*
   *
   * cv::MserFeatureDetector::MserFeatureDetector (
   *            int     delta, //in the code, it compares (size_{i}-size_{i-delta})/size_{i-delta} 
   *            int     minArea, //prune the area which smaller than minArea 
   *            int     maxArea, //prune the area which bigger than maxArea 
   *            double  maxVariation, //prune the area have simliar size to its children 
   *            double  minDiversity, //trace back to cut off mser with diversity < min_diversity 
   *            int     maxEvolution, //for color image, the evolution steps 
   *            double  areaThreshold, //the area threshold to cause re-initialize 
   *            double  minMargin,  //ignore too small margin 
   *            int     edgeBlurSize //the aperture size for edge blur 
   *    )       
   */
MserFeatureDetector detector;

BriefDescriptorExtractor descriptor(64);



/// Function header
void interface(int argc, void *);

/**
 * @function main
 */
int main(int, char **argv)
{

    bool aCorres = false;

    image1 = imread(argv[1], 1);
    image2 = imread(argv[2], 1);
    rows = image1.rows;
    cols = image1.cols;


    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);



    detector.detect(image1, keypoints1);
    detector.detect(image2, keypoints2);


    Mat descriptors1, descriptors2;

    descriptor.compute(image1, keypoints1, descriptors1);
    descriptor.compute(image2, keypoints2, descriptors2);



    // Construction of the matcher
    //BruteForceMatcher< HammingLUT > matcher;
    BruteForceMatcher < Hamming > matcher;	// =BruteForceMatcher<Hamming>(10);

    Mat descriptorAuxKp1;
    Mat descriptorAuxKp2;

    Mat descriptorVois;

    vector < int >associateIdx;
    vector < int >associateIdxVois;

    vector < KeyPoint > keypointsVois;
    vector < KeyPoint > pointsx;
    vector < KeyPoint > pointsy;


    /*nb de points reliés */
    int pointCourant = 0;

    for (int i = 0; i < keypoints1.size(); i++) {
	//on a pas encore trouvé de correspondant pour kp1
	aCorres = false;
	//on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le Keypoints[i]
	descriptors1.row(i).copyTo(descriptorAuxKp1);

//ici on va mettre que les valeurs du descripteur des keypoints de l'image 2 que l'on veut comparer aux keypoints de l'image1 en cours de traitement
	descriptorAuxKp2.create(0, 0, CV_8UC1);

	keypointsVois.erase(keypointsVois.begin(), keypointsVois.end());
	for (int j = 0; j < keypoints2.size(); j++) {

	    float p1x = keypoints1[i].pt.x;
	    float p1y = keypoints1[i].pt.y;
	    float p2x = keypoints2[j].pt.x;
	    float p2y = keypoints2[j].pt.y;

	    float distance = sqrt(pow((p1x - p2x), 2) + pow((p1y - p2y), 2));

	    //parmis les valeurs dans descriptors2 on ne va garder que ceux dont les keypoints associés sont à une distance définie du keypoints en cours, en l'occurence le ieme ici.
	    if (distance < 4) {
		aCorres = true;

		KeyPoint kpVois;

                


		for (int ivois = -1; ivois < 2; ivois++) {

		    for (int jvois = -1; jvois < 2; jvois++) {

			kpVois.pt.x = p2x + ivois;
			kpVois.pt.y = p2y + jvois;
                        kpVois.size = keypoints2[0].size;
                       // keypointsVois.push_back(kpVois);

			/*il faut que les coordonnes du point en question soient contenus dans l'image */
			if (kpVois.pt.x >= 0 && kpVois.pt.x < image2.rows
			    && kpVois.pt.y >= 0 && kpVois.pt.y < image2.cols) {

			    keypointsVois.push_back(kpVois);

			}
		    }
		}


		/*on calcule les descripteurs des keypointsVois */
		descriptor.compute(image2, keypointsVois, descriptorVois);

		/*         float pxvois=keypoints1[i].pt.x;
		   float pyvois=keypoints1[i].pt.y;
		   float p2xvois=keypoints2[j].pt.x;
		   float p2yvois=keypoints2[j].pt.y; */


		for (int idxVois = 0; idxVois < descriptorVois.rows; idxVois++) {
		    descriptorAuxKp2.push_back(descriptorVois.row(idxVois));
		}

		//descriptorAuxKp2.push_back(descriptors2.row(j));
		//associateIdx.push_back(j);



	    }			//fin de if(distance<4)




	}			//fin for(int j=0;j<descriptors2.rows;j++)


	//il faudra vérifier que le kp1 a un kp2 possible
	if (aCorres) {
	    pointsx.push_back(keypoints1[i]);




	    //ici on ne matche qu'un keypoints de l'image1 avec le meilleur des keypoints gardés de l'image 2
	    matcher.match(descriptorAuxKp1, descriptorAuxKp2, matches);

	    //on a trouvé le keypoints qui va le mieux
	    //nrmlt on a trouvé un kp

	    cout << "\n On a trouvé " << matches.size() << " points correspondants";
	    KeyPoint best2 = keypointsVois[matches[0].trainIdx];

	    pointsy.push_back(best2);

	    matches[0].trainIdx = pointCourant;
	    matches[0].queryIdx = pointCourant;
            pointCourant++;
	    matchesWithDist.insert(matchesWithDist.end(), matches.begin(), matches.end());


	}




    }				//fin du for i



//ici on trie les matchesWithDist par distance des valeurs des descripteurs et non par distance euclidienne
    nth_element(matchesWithDist.begin(), matchesWithDist.begin() + 24, matchesWithDist.end());
    // initial position
    // position of the sorted element
    // end position

    Mat imageMatches;
    Mat matchesMask;
    drawMatches(image1, pointsx,	// 1st image and its keypoints
		image2, pointsy,	// 2nd image and its keypoints
		matchesWithDist,	// the matches
		imageMatches,	// the image produced
		Scalar::all(-1),	// color of the lines
		Scalar(255, 255, 255)	//color of the keypoints
	);


    namedWindow(matches_window, CV_WINDOW_AUTOSIZE);
    imshow(matches_window, imageMatches);
    imwrite("resultat.png", imageMatches);



    /// Create a window and a trackbar
    namedWindow(transparency_window, WINDOW_AUTOSIZE);
    createTrackbar("Threshold: ", transparency_window, &thresh, max_thresh, interface);
    //imshow(transparency_window,image1 );







    interface(0, 0);

    waitKey(0);
    return (0);
}

/**
 * @function cornerHarris_demo
 * @brief Executes the corner detection and draw a circle around the possible corners
 */
void interface(int, void *)
{

    Mat dst;
    image1.copyTo(dst);

    ///on adapte l'importance des pixels de chaque image selon la valeur du trackbar
    for (int i = 0; i < rows; i++) {
	for (int j = 0; j < cols; j++) {

	    dst.at < cv::Vec3b > (i, j)[0] = (float) (image2.at < cv::Vec3b > (i, j)[0]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[0]) * (float) ((100. - thresh) / 100.);
	    dst.at < cv::Vec3b > (i, j)[1] = (float) (image2.at < cv::Vec3b > (i, j)[1]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[1]) * (float) ((100. - thresh) / 100.);
	    dst.at < cv::Vec3b > (i, j)[2] = (float) (image2.at < cv::Vec3b > (i, j)[2]) * (float) (thresh / 100.) + (float) (image1.at < cv::Vec3b > (i, j)[2]) * (float) ((100. - thresh) / 100.);


	}
    }

    float kp1x;
    float kp1y;
    float kp2x;
    float kp2y;

    float kptx;
    float kpty;

    vector < KeyPoint > keypointsKeep1, keypointsKeep2;


    for (int i = 0; i < matchesWithDist.size(); i++) {


	kp1x = keypoints1[matchesWithDist[i].queryIdx].pt.x;
	kp1y = keypoints1[matchesWithDist[i].queryIdx].pt.y;
	kp2x = keypoints2[matchesWithDist[i].trainIdx].pt.x;
	kp2y = keypoints2[matchesWithDist[i].trainIdx].pt.y;

	kptx = kp1x * (100. - thresh) / 100. + kp2x * (thresh / 100.);
	kpty = kp1y * (100. - thresh) / 100. + kp2y * (thresh / 100.);

	Point ptkp1 = Point(kptx, kpty);

	int nbColor = 256 * 256 * 256;

	int pascoul = nbColor / matchesWithDist.size();
	int coulActu = pascoul * i;

	int bleu = coulActu % 256;

	int qb = coulActu / 256;
	int vert = qb % 256;

	int rouge = qb / 256;



	circle(dst, ptkp1, 5, Scalar(rouge, vert, bleu), 2, 8, 0);
    }

    namedWindow(transparency_window, WINDOW_AUTOSIZE);
    imshow(transparency_window, dst);
}

