/**
 * il faut configurer Harris avec
 * int blockSize;
 * int apertureSize;
 * double k;
 *
 * et il faut mettre un size aux keypoints (e.g 100.0)
 *
 * Il faut configurer BriefDescriptorExtractor
 *
 * et Bruteforce
 * configurer la distance correspondant aux voisins autour desquels on cherche
 */








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

const char *transparency_window = "transparence Harris+BRIEF+FLANN";
const char *matches_window = "Harris+BRIEF+FLANN";

int thresh = 0;
int max_thresh = 100;

/// variables for sift

vector < KeyPoint > keypoints1, keypoints2;
  //vector<DMatch>  matches;
vector < DMatch > matches;

  //vector<vector<DMatch> > matchesWithDist;
vector < DMatch > matchesWithDist;

/// Function header
void interface(int argc, void *);

/**
 * @function main
 */
int main(int, char **argv)
{



    image1 = imread(argv[1], 1);
    image2 = imread(argv[2], 1);
    rows = image1.rows;
    cols = image1.cols;


    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);

    Mat image1_gray;
    Mat image2_gray;


    /// Converts an image from one color space to another.
    cvtColor(image1, image1_gray, COLOR_BGR2GRAY);
    cvtColor(image2, image2_gray, COLOR_BGR2GRAY);

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    /*
       void ocl::cornerHarris(const oclMat& src, oclMat& dst, int blockSize, int ksize, double k, int bordertype=cv::BORDER_DEFAULT)

       src – Source image. Only CV_8UC1 and CV_32FC1 images are supported now.
       dst – Destination image containing cornerness values. It has the same size as src and CV_32FC1 type.
       blockSize – Neighborhood size
       ksize – Aperture parameter for the Sobel operator
       k – Harris detector free parameter
       bordertype – Pixel extrapolation method. Only BORDER_REFLECT101, BORDER_REFLECT, BORDER_CONSTANT and BORDER_REPLICATE are supported now.
     */

    Mat image1dst;
    Mat image2dst;

    image1dst = Mat::zeros(image1.size(), CV_32FC1);
    image2dst = Mat::zeros(image2.size(), CV_32FC1);


    cornerHarris(image1_gray, image1dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    cornerHarris(image2_gray, image2dst, blockSize, apertureSize, k, BORDER_DEFAULT);

    int threshHarris = 100;

    /// Normalizing
    /*
       void normalize(InputArray src, OutputArray dst, double alpha=1, double beta=0, int norm_type=NORM_L2, int dtype=-1, InputArray mask=noArray() )
       src – input array.
       dst – output array of the same size as src .
       alpha – norm value to normalize to or the lower range boundary in case of the range normalization.
       beta – upper range boundary in case of the range normalization; it is not used for the norm normalization.
       normType – normalization type (see the details below).
       dtype – when negative, the output array has the same type as src; otherwise, it has the same number of channels as src and the depth =CV_MAT_DEPTH(dtype).
       mask – optional operation mask.
     */

    Mat image1dst_norm;
    Mat image2dst_norm;

    normalize(image1dst, image1dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    normalize(image2dst, image2dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    /*
       On each element of the input array, the function convertScaleAbs performs three operations sequentially: scaling, taking an absolute value, conversion to an unsigned 8-bit type:

     */

    Mat image1dst_norm_scaled;
    Mat image2dst_norm_scaled;

    convertScaleAbs(image1dst_norm, image1dst_norm_scaled);
    convertScaleAbs(image2dst_norm, image2dst_norm_scaled);

    KeyPoint kp;

    for (int j = 0; j < image1dst_norm.rows; j++) {
	for (int i = 0; i < image1dst_norm.cols; i++) {
	    if ((int) image1dst_norm.at < float >(j, i) > threshHarris) {

		kp.pt.x = (float) j;
		kp.pt.y = (float) i;
		//necessaire je ne sais pas pk
		kp.size = 100.0;
		keypoints1.push_back(kp);

	    }
	}
    }

    for (int j = 0; j < image2dst_norm.rows; j++) {
	for (int i = 0; i < image2dst_norm.cols; i++) {
	    if ((int) image2dst_norm.at < float >(j, i) > threshHarris) {


		kp.pt.x = (float) j;
		kp.pt.y = (float) i;
		//necessaire je ne sais pas pk
		kp.size = 100.0;
		keypoints2.push_back(kp);

	    }
	}
    }


    BriefDescriptorExtractor briefDesc(64);

    Mat descriptors1, descriptors2;
    briefDesc.compute(image1, keypoints1, descriptors1);
    briefDesc.compute(image2, keypoints2, descriptors2);

    //Ptr<DescriptorMatcher> matcher =  new FlannBasedMatcher(DescriptorMatcher::create("Flann"));
    FlannBasedMatcher matcher;

    Mat descriptorAuxKp1;
    Mat descriptorAuxKp2;


    vector < int >associateIdx;

    for (int i = 0; i < descriptors1.rows; i++) {
	//on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le Keypoints[i]
	descriptors1.row(i).copyTo(descriptorAuxKp1);

//ici on va mettre que les valeurs du descripteur des keypoints de l'image 2 que l'on veut comparer aux keypoints de l'image1 en cours de traitement
	descriptorAuxKp2.create(0, 0, CV_8UC1);


	//associateIdx va servir à faire la transition entre les indices renvoyés par matches et ceux des Keypoints
	associateIdx.erase(associateIdx.begin(), associateIdx.end());


	for (int j = 0; j < descriptors2.rows; j++) {

	    float p1x = keypoints1[i].pt.x;
	    float p1y = keypoints1[i].pt.y;
	    float p2x = keypoints2[j].pt.x;
	    float p2y = keypoints2[j].pt.y;

	    float distance = sqrt(pow((p1x - p2x), 2) + pow((p1y - p2y), 2));

	    //parmis les valeurs dans descriptors2 on ne va garder que ceux dont les keypoints associés sont à une distance définie du keypoints en cours, en l'occurence le ieme ici.
	    if (distance < 10) {

		descriptorAuxKp2.push_back(descriptors2.row(j));
		associateIdx.push_back(j);

	    }


	}
	//ici on ne matche qu'un keypoints de l'image1 avec tous les keypoints gardés de l'image 2
        matcher.add(descriptorAuxKp1);
        matcher.train();

	matcher.match(descriptorAuxKp2, matches);

	//on remet à la bonne valeur les attributs de matches
	for (int idxMatch = 0; idxMatch < matches.size(); idxMatch++) {
	    //on a comparer le keypoints i
	    matches[idxMatch].queryIdx = i;
	    //avec le keypoints2 j
	    matches[idxMatch].trainIdx = associateIdx[matches[idxMatch].trainIdx];
	}

	//on concatene les matches trouvés pour les points précedents avec les nouveaux
	matchesWithDist.insert(matchesWithDist.end(), matches.begin(), matches.end());


    }



//ici on trie les matchesWithDist par distance des valeurs des descripteurs et non par distance euclidienne
    nth_element(matchesWithDist.begin(), matchesWithDist.begin() + 24, matchesWithDist.end());
    // initial position
    // position of the sorted element
    // end position

    Mat imageMatches;
    Mat matchesMask;
    drawMatches(image1, keypoints1,	// 1st image and its keypoints
		image2, keypoints2,	// 2nd image and its keypoints
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

