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


string transparency_window;
string matches_window;



int thresh = 0;
int max_thresh = 100;


//seuil pour la comparaison des descripteurs
float seuil=85;

//rayon autour duquel on cherche les pixels autours d'un kp1
int rayonDist = 5;

//dans keypoints1 on a les kp de l'image de gauche auxquels 
//on a apparié le meilleur pixel du voisinage correspondant

//on a appariés les pointsx de l'image gauche avec les pointsy de l'image de droite

vector < KeyPoint > keypoints1, pointsx, pointsy;

//puis on a vérifies qu'en appariant les pointsy avec l'image gauche, on retombait sur 
//nos pas, ce qui a formé pointsy2 et pointsx2

vector < KeyPoint > pointsy2,pointsx2; 

//les matches effectués (variable auxiliaire)
vector < DMatch > matches;

//matches en prenant en compte la distance
vector < DMatch > matchesWithDist,matchesWithDist2;

/****************************/
/*                          */
/*      LES DETECTEURS      */
/*                          */
/****************************/

//utilisation de MSER
/*
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

//utilisation de FAST

/* cv::FastFeatureDetector::FastFeatureDetector (
  *	int threshold = 10,
  *	bool nonmaxSuppression = true
  *	)
  */

//utilisation de STAR

/*
*cv::StarFeatureDetector::StarFeatureDetector (
	int maxSize,
	int responseThreshold = 30,
	int lineThresholdProjected = 10,
	int lineThresholdBinarized = 8,
	int suppressNonmaxSize = 5
   )
   */

//utilisation de SIFT
/*
cv::SiftFeatureDetector::SiftFeatureDetector (
	double  threshold,
	double  edgeThreshold,
	int nOctaves = SIFT::CommonParams::DEFAULT_NOCTAVES,
	int nOctaveLayers = SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS,
	int firstOctave = SIFT::CommonParams::DEFAULT_FIRST_OCTAVE,
	int angleMode = SIFT::CommonParams::FIRST_ANGLE
  )
*/

//utilisation de SURF
/*
cv::SurfFeatureDetector::SurfFeatureDetector (
        double hessianThreshold = 400.,
	int octaves = 3,
	int octaveLayers = 4
	)
*/

//utilisation de ORB
/*
ORB::ORB(
int nfeatures=500,
float scaleFactor=1.2f,
int nlevels=8,
int edgeThreshold=31,
int firstLevel=0,
int WTA_K=2,
int scoreType=ORB::HARRIS_SCORE,
int patchSize=31)
*/

//utilisation de HARRIS avec GoodFeaturesToTrackDetector
/*
 	GoodFeaturesToTrackDetector (
         int maxCorners,
         double qualityLevel,
         double minDistance,
         int blockSize=3,
         true,
         double k=0.04)
*/

//utilisation de GFTT i.e cornerMinEigenVal, les minimum des valeurs propres des vecteurs.
/*
 	GoodFeaturesToTrackDetector (
         int maxCorners,
         double qualityLevel,
         double minDistance,
         int blockSize=3,
         bool useHarrisDetector=false,
         double k=0.04)
*/

//utilisation de DENSE

/*

cv::DenseFeatureDetector::Params::Params (
float  	initFeatureScale = 1.f,
int  	featureScaleLevels = 1,
float  	featureScaleMul = 0.1f,
int  	initXyStep = 6,
int  	initImgBound = 0,
bool  	varyXyStepWithScale = true,
bool  	varyImgBoundWithScale = false
)

*/

//utilisation de SIMPLE BLOB

/*

class SimpleBlobDetector : public FeatureDetector
{
public:
struct Params
{
    Params();
    float thresholdStep;
    float minThreshold;
    float maxThreshold;
    size_t minRepeatability;
    float minDistBetweenBlobs;

    bool filterByColor;
    uchar blobColor;

    bool filterByArea;
    float minArea, maxArea;

    bool filterByCircularity;
    float minCircularity, maxCircularity;

    bool filterByInertia;
    float minInertiaRatio, maxInertiaRatio;

    bool filterByConvexity;
    float minConvexity, maxConvexity;
};

SimpleBlobDetector(const SimpleBlobDetector::Params ¶meters = SimpleBlobDetector::Params());

*/















 /**************************************/
 /*                                    */
 /*          LES DESCRIPTEURS          */
 /*                                    */
 /**************************************/



//SIFT

/*
 * SiftDescriptorExtractor (
     double magnification, 
     bool isNormalize = true, 
     bool recalculateAngles = true,
     int nOctaves=SIFT::CommonParams::DEFAULT_NOCTAVES, 
     int nOctaveLayers=SIFT::CommonParams::DEFAULT_NOCTAVE_LAYERS, 
     int firstOctave=SIFT::CommonParams::DEFAULT_FIRST_OCTAVE, 
     int angleMode=SIFT::CommonParams::FIRST_ANGLE
)
 *
 *mais on a:
 *SIFT(int nfeatures=0,
       int nOctaveLayers=3,
       double contrastThreshold=0.04,
       double edgeThreshold=10, 
       double sigma=1.6) 
 */

//SURF

/*
 * SurfDescriptorExtractor (int nOctaves=4, int nOctaveLayers=2, bool extended=false)
 * 
 * mais on a:
 *
 *SURF(double hessianThreshold, int nOctaves=4, int nOctaveLayers=2, bool extended=false, bool upright=false)
 */


//BRIEF
/*
 *
 *cv::BriefDescriptorExtractor::BriefDescriptorExtractor ( int bytes = 32 ) 
 *
 * // bytes is a length of descriptor in bytes. It can be equal 16, 32 or 64 bytes. 
 */
//ORB
/*
 *OrbDescriptorExtractor( ORB::PatchSize patch_size );
 *
 */




 /**************************************/
 /*                                    */
 /*           LES APPARIEURS           */
 /*                                    */
 /**************************************/

/*apparieurs fonctionnant avec les descripteurs SIFT et SURF*/

//BruteForce

//BruteForce-L1


/*apparieurs fonctionnant avec les descripteurs BRIEF et ORB */

//BruteForce-Hamming

//FlannBased



/// Function header
void interface(int argc, void *);


string type2str(int type);

//fonction pour connaitre le type de donnees dans une matrice
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

/**
 * @function main
 */



int main(int argc, char **argv)
{


    Ptr < DescriptorExtractor > descriptor;
    Ptr < DescriptorMatcher > matcher;



    if (argc != 6) {
	cout << "\nusage incorrect!nn";
	cout << "pathimage1 pathimage2 detectorName descriptorName matcherNamen" << endl;

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
	cout << "\n SIMPLE BLOBn";
	cout << endl;

	cout << "\n descriptorName List:";
	cout << "\n SIFT";
	cout << "\n SURF";
	cout << "\n BRIEF";
	cout << "\n ORB";
	cout << endl;

	cout << "\n matcherName List:";

	cout << "\n\n pour les descripteurs SIFT et SURFn";
	cout << "\n BruteForce";
	cout << "\n BruteForce-L1";

	cout << "\n\n pour les descripteurs BRIEF et ORBn";
	cout << "\n BruteForce-Hamming";
	cout << "\n FlannBased";
	cout << "\n";
	cout << endl;


	exit(0);
    }



    image1 = imread(argv[1], 1);
    image2 = imread(argv[2], 1);



    //selection et utilisation du detecteur

    if (strcmp(argv[3], "MSER") == 0) {
	/*
	   MSER* detector = new MSER();

	   //(*detector)(image1,Mat(),keypoints1);
	   //(*detector)(image2,Mat(),keypoints2);
	 */

	MserFeatureDetector detector;
	detector.detect(image1, keypoints1);


    }


    else if (strcmp(argv[3], "FAST") == 0) {

	FastFeatureDetector detector(50, true);
	detector.detect(image1, keypoints1);


    }


    else if (strcmp(argv[3], "STAR") == 0) {


	StarFeatureDetector detector;
	detector.detect(image1, keypoints1);


    }


    else if (strcmp(argv[3], "SIFT") == 0) {

	SiftFeatureDetector detector;
	detector.detect(image1, keypoints1);

	/*
	   SIFT detector;
	   detector(image1,Mat(),keypoints1);
	 */

    }

    else if (strcmp(argv[3], "SURF") == 0) {

	SurfFeatureDetector detector;
	detector.detect(image1, keypoints1);

	/*
	   SURF detector(100);
	   detector(image1,Mat(),keypoints1);
	 */

    }
    //maniere differente d'implementer
    else if (strcmp(argv[3], "ORB") == 0) {
	ORB detector(200);
	detector(image1, Mat(), keypoints1);

	/*
	   OrbFeatureDetector detector;
	   detector.detect(image1, keypoints1);
	 */
    }


    else if (strcmp(argv[3], "HARRIS") == 0) {
	GoodFeaturesToTrackDetector detector(1000, 0.01, 1., 3, true, 0.04);
	detector.detect(image1, keypoints1);


    } else if (strcmp(argv[3], "GFTT") == 0) {
	GoodFeaturesToTrackDetector detector;
	detector.detect(image1, keypoints1);

    }

    else if (strcmp(argv[3], "DENSE") == 0) {
	Ptr < FeatureDetector > detector;
	detector = FeatureDetector::create("Dense");

	//DenseFeatureDetector detector(0.5);
	detector->detect(image1, keypoints1);

    } else if (strcmp(argv[3], "SIMPLE BLOB") == 0) {
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 10.0;	// minimum 10 pixels between blobs
	params.filterByArea = true;	// filter my blobs by area of blob
	params.minArea = 10.0;	// min 20 pixels squared
	params.maxArea = 500.0;	// max 500 pixels squared
	SimpleBlobDetector detector(params);



	detector.detect(image1, keypoints1);

    }


    else {

	cout << "\n detecteur inconnu" << endl;
	cout << "\n liste des détecteurs possible:";
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
	cout << "\n";
	exit(0);
    }


    //selection du descripteur



    if (strcmp(argv[4], "BRIEF") == 0) {

	//descriptor = cv::DescriptorExtractor::create("BRIEF");
	descriptor = new BriefDescriptorExtractor(64);


    } else if (strcmp(argv[4], "ORB") == 0) {
	//descriptor = cv::DescriptorExtractor::create("ORB");
	descriptor = new OrbDescriptorExtractor();

    } else if (strcmp(argv[4], "SIFT") == 0) {
	descriptor = new SiftDescriptorExtractor();

	//descriptor = cv::DescriptorExtractor::create("SIFT");

    } else if (strcmp(argv[4], "SURF") == 0) {

	//descriptor = cv::DescriptorExtractor::create("SURF");

	descriptor = new SurfDescriptorExtractor();

    }

    /*else if(strcmp(argv[4],"CALONDER")==0){

       //descriptor = cv::DescriptorExtractor::create("SURF");

       descriptor = new CalonderDescriptorExtractor<float>();

       }
     */



    else {

	cout << "\n descripteur inconnu" << endl;
	cout << "\n liste des descripteurs possible:";
	cout << "\n SIFT";
	cout << "\n SURF";
	cout << "\n BRIEF";
	cout << "\n ORB";
	cout << "\n";

	exit(0);
    }

    if (strcmp(argv[5], "BruteForce") == 0 || strcmp(argv[5], "BruteForce-L1") == 0 || strcmp(argv[5], "BruteForce-Hamming") == 0 || strcmp(argv[5], "FlannBased") == 0) {
	matcher = cv::DescriptorMatcher::create(argv[5]);



    }

    else {
	cout << "\n apparieur inconnu" << endl;
	cout << "\n liste des apparieurs possible:";
	cout << "\n BruteForce";
	cout << "\n BruteForce-L1";
	cout << "\n BruteForce-Hamming";
	cout << "\n FlannBased";
	cout << "\n";

	exit(0);


    }



    //chaine da caractère constitué du detecteur, du descripteur et de l'extracteur
    string st1 = argv[3] + (string) " + " + argv[4] + (string) " + " + argv[5];
    string st2 = "transparence " + st1;

    transparency_window = st2;
    matches_window = st1;



    rows = image1.rows;
    cols = image1.cols;


    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);


    


	//matrice des descripteurs des kp1
    Mat descriptors1;

    (*descriptor).compute(image1, keypoints1, descriptors1);

    cout << endl <<"nb de kp:"<< keypoints1.size()<<endl;

    //matrice auxiliaire des descripteurs d'un Kp
    Mat descriptorAuxKp1;
    //descriptorAuxKp1.create(0, 0, CV_32FC1);

    //matrice des descripteurs du voisinage du Kp étudié
    Mat descriptorVois;
    //descriptorVois.create(0, 0, CV_32FC1);
     

    //pixels du voisinage d'un Kp
    vector < KeyPoint > pixelsVois;


    //si on a trouvé un keypoint2 de l'image2 correspondant au keypoint1 de l'image 1 en cours
    bool aCorres = false;
    //nb de points reliés
    int pointCourant = 0;

    for (int i = 0; i < keypoints1.size(); i++) {
	//on a pas encore trouvé de correspondant pour kp1
	aCorres = false;
	//on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le Keypoints[i]



       // descriptorAuxKp1.convertTo(descriptorAuxKp1, CV_8UC1);
	descriptors1.row(i).copyTo(descriptorAuxKp1);

        

//on supprime tous les pixels associés au keypoint1 precedent
	pixelsVois.erase(pixelsVois.begin(), pixelsVois.end());

	float p1x = keypoints1[i].pt.x;
	float p1y = keypoints1[i].pt.y;

	for (int x2 = -rayonDist; x2 < rayonDist + 1; x2++) {
	    for (int y2 = -rayonDist; y2 < rayonDist + 1; y2++) {

		float p2x = p1x + x2;
		float p2y = p1y + y2;

		float distance = sqrt(pow(x2, 2) + pow(y2, 2));

		//il faut que les coordonnes du point en question soient contenus dans l'image

		//verifier si rows et cols sont pas inverser
		if (p2x >= 0 && p2x < image1.rows && p2y >= 0 && p2y < image1.cols) {

		    //on ne va garder que les pixels situés à une distance rayondist du point d'intérêt Pil(i)
             
		    if (distance <= rayonDist) {
			//on a au moins un keypoint correspondant
			aCorres = true;

			KeyPoint kpVois;
			kpVois.pt.x = p2x;
			kpVois.pt.y = p2y;
			kpVois.size = keypoints1[0].size;

			pixelsVois.push_back(kpVois);
		    }

		}


	    }
	}



	//il faudra vérifier que le kp1 a un kp2 possible
	if (aCorres) {
	    //on sait que le keypoints1[i] va être relié
	    pointsx.push_back(keypoints1[i]);

	    //on calcule le descripteur de tous les pixels retenus comme candidat

	    (*descriptor).compute(image2, pixelsVois, descriptorVois);



	    //ici on ne matche qu'un keypoints de l'image1 avec le meilleur des keypoints gardés de l'image 2

	    //affiche les descripteurs

	    //cout << "type" << descriptorAuxKp1.type();
    /*descriptorAuxKp1.convertTo(descriptorAuxKp1, CV_32FC1);
    descriptorVois.convertTo(descriptorVois, CV_32FC1);*/

   // descriptorAuxKp1.convertTo(descriptorAuxKp1, CV_32FC1);
/*
    string ty = type2str(descriptorAuxKp1.type());
    string ty2 = type2str(descriptorVois.type());


    printf("Matrix: %s %dx%d \n", ty.c_str(), descriptorAuxKp1.cols, descriptorAuxKp1.rows);
    printf("Matrix: %s %dx%d \n", ty.c_str(), descriptorVois.cols, descriptorVois.rows);
*/           

            
            //cout << descriptorAuxKp1;
	    matcher->match(descriptorAuxKp1, descriptorVois, matches);

	    //on a trouvé le keypoints qui va le mieux
	    //nrmlt on a trouvé un kp


	    KeyPoint best2 = pixelsVois[matches[0].trainIdx];

	    pointsy.push_back(best2);

	    matches[0].trainIdx = pointCourant;
	    matches[0].queryIdx = pointCourant;



	    pointCourant++;
	    matchesWithDist.insert(matchesWithDist.end(), matches.begin(), matches.end());


	}


    }				//fin du for i

//on repart dans l'autre sens. Les pixels sélectionnés sont dans pointsy









/**********************************************************************************************/

pointCourant = 0;

//la matrice des descripteurs des pixels gardés
 Mat descriptorsPixs;
//attention on calcule les descripteurs de l'image 2
 (*descriptor).compute(image2, pointsy, descriptorsPixs);

 for (int i = 0; i < pointsy.size(); i++) {
	//on a pas encore trouvé de correspondant pour pointsy[i]
	aCorres = false;
	//on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le pointsy[i]
	descriptorsPixs.row(i).copyTo(descriptorAuxKp1);



//on supprime tous les pixels associés au keypoint1 precedent
	pixelsVois.erase(pixelsVois.begin(), pixelsVois.end());

	float p1x = pointsy[i].pt.x;
	float p1y = pointsy[i].pt.y;
/*
        cout << "\n le point x " <<p1x <<" y "<<p1y;
        cout << "\n rows1 " << image1.rows;    
        cout << "\n cols1 " << image1.cols;  
        cout << "\n rows2 " << image2.rows;    
        cout << "\n cols2 " << image2.cols;     
*/

	for (int x2 = -rayonDist; x2 < rayonDist + 1; x2++) {
	    for (int y2 = -rayonDist; y2 < rayonDist + 1; y2++) {

		float p2x = p1x + x2;
		float p2y = p1y + y2;

		float distance = sqrt(pow(x2, 2) + pow(y2, 2));

		//il faut que les coordonnes du point en question soient contenus dans l'image

		//verifier si rows et cols sont pas inversee
		if (p2x >= 0 && p2x < image2.rows && p2y >= 0 && p2y < image2.cols) {

		    //on ne va garder que les pixels situés à une distance rayondist de pointsy[i]

		    if (distance <= rayonDist) {
			//on a au moins un keypoint correspondant
			aCorres = true;

			KeyPoint kpVois;
			kpVois.pt.x = p2x;
			kpVois.pt.y = p2y;
			kpVois.size = keypoints1[0].size;

			pixelsVois.push_back(kpVois);
		    }

		}


	    }
	}



	//il faudra vérifier que le kp2 a des pixels correspondant possible
	if (aCorres){

	    //on calcule le descripteur de tous les pixels retenus comme candidat
            //attention ce sont les descripteurs de image1*/

	    (*descriptor).compute(image1, pixelsVois, descriptorVois);

	    //ici on ne matche qu'un keypoints de l'image2 avec le meilleur des pixels gardés de l'image 1

	    //affiche les descripteurs

	    //cout << "type" << descriptorAuxKp1.type();


   /* string ty = type2str(descriptorAuxKp1.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), descriptorAuxKp1.cols, descriptorAuxKp1.rows);
*/
            
            

            /*descriptorAuxKp1.convertTo(descriptorAuxKp1, CV_32FC1);
            descriptorVois.convertTo(descriptorVois, CV_32FC1);*/


	    matcher->match(descriptorAuxKp1, descriptorVois, matches);

	    //on a trouvé le keypoints qui va le mieux
	    //nrmlt on a trouvé un kp


	    KeyPoint best3 = pixelsVois[matches[0].trainIdx];

	    //il nous faut comparer les points

            //if(best3.pt.x == pointsx[i].pt.x && best3.pt.y == pointsx[i].pt.y){
                //si on est retombé sur nos pas
            if( (abs( (best3.pt.x) - (pointsx[i].pt.x) ) <= 0.) && (abs ( (best3.pt.y) - pointsx[i].pt.y) <= 0.)){

                
                matches[0].trainIdx = pointCourant;
	    	matches[0].queryIdx = pointCourant;

		//cout << endl <<matches[0].distance;
                if(abs(matches[0].distance)< seuil){
                cout << matches[0].distance;
                 
                pointsx2.push_back(best3);
                pointsy2.push_back(pointsy[i]);
                
                matchesWithDist2.insert(matchesWithDist2.end(), matches.begin(), matches.end());
                pointCourant++;
                }


            }            

            /*
	    pointsz.push_back(best3);
           
	    matches[0].trainIdx = pointCourant;
	    matches[0].queryIdx = pointCourant;
            */


	    //pointCourant++;
	   // matchesWithDist.insert(matchesWithDist.end(), matches.begin(), matches.end());


	}
        else{
               
		cout << "\n Houston on a un gros probleme \n";
                 cout << "\n au point x " <<p1x <<" y "<<p1y;
        }


    }				//fin du for i


/*********************************************************************************************/














//ici on trie les matchesWithDist2 par distance des valeurs des descripteurs et non par distance euclidienne
    nth_element(matchesWithDist2.begin(), matchesWithDist2.begin() + 24, matchesWithDist2.end());
    // initial position
    // position of the sorted element
    // end position

    //si on veut garder les meilleurs correspondances
/*
     if(matchesWithDist.size()>500){
       matchesWithDist.erase(matchesWithDist.begin() + 500,matchesWithDist.end());
     }
*/

    Mat imageMatches;
    Mat matchesMask;
    drawMatches(image1, pointsx2,	// 1st image and its keypoints
		image2, pointsy2,	// 2nd image and its keypoints
		matchesWithDist2,	// the matches
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


    for (int i = 0; i < matchesWithDist2.size(); i++) {


	kp1x = pointsx2[matchesWithDist2[i].queryIdx].pt.x;
	kp1y = pointsx2[matchesWithDist2[i].queryIdx].pt.y;
	kp2x = pointsy2[matchesWithDist2[i].trainIdx].pt.x;
	kp2y = pointsy2[matchesWithDist2[i].trainIdx].pt.y;

	kptx = kp1x * (100. - thresh) / 100. + kp2x * (thresh / 100.);
	kpty = kp1y * (100. - thresh) / 100. + kp2y * (thresh / 100.);

	Point ptkp1 = Point(kptx, kpty);

	int nbColor = 256 * 256 * 256;

	int pascoul = nbColor / matchesWithDist2.size();
	int coulActu = pascoul * i;

	int bleu = coulActu % 256;

	int qb = coulActu / 256;
	int vert = qb % 256;

	int rouge = qb / 256;



	circle(dst, ptkp1, 5, Scalar(rouge, vert, bleu), 2, 8, 0);
    }

    namedWindow(transparency_window, WINDOW_AUTOSIZE);
    imshow(transparency_window, dst);
    imwrite("trans.png", dst);
}

