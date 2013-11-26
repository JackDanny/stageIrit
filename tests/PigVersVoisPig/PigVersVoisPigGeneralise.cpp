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


vector < KeyPoint > keypoints1, keypoints2, pointsx, pointsy;
//vector<DMatch>  matches;
vector < DMatch > matches;

//vector<vector<DMatch> > matchesWithDist;
vector < DMatch > matchesWithDist;


//LES DETECTEURS

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

//utilisation de GFTT
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

SimpleBlobDetector(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

*/























//Les descripteurs

//cv::Ptr<cv::DescriptorExtractor> descriptor =cv::Algorithm::create<cv::DescriptorExtractor>("Feature2D.BRISK");





//descriptor



/// Function header
void interface(int argc, void *);

/**
 * @function main
 */


int main(int argc, char **argv)
{

   // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SURF");
    //Ptr<DescriptorExtractor> descriptor = new SiftDescriptorExtractor(40);
    cv::Ptr<cv::DescriptorExtractor> descriptor;  
   
    if(argc!=5)
    {
        cout << "\nusage incorrect!\n\n";
        cout << "pathimage1 pathimage2 detectorName descriptorName\n" <<endl;

        cout << "\n detectorNameList:";
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

        exit(0);
    }



    image1 = imread(argv[1], 1);
    image2 = imread(argv[2], 1);
    String detectorName=argv[3];
    //cout << "\n nom du détecteur: "<<argv[3];

    if(strcmp(argv[3],"MSER")==0)
    {
        MserFeatureDetector detector;
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);


    }


    else if(strcmp(argv[3],"FAST")==0)
    {
        FastFeatureDetector detector(50,true);
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);
    }


    else if(strcmp(argv[3],"STAR")==0)
    {
        StarFeatureDetector detector;
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);


    }


    else if(strcmp(argv[3],"SIFT")==0)
    {
        SiftFeatureDetector detector;
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);


    }

    else if(strcmp(argv[3],"SURF")==0)
    {
        SurfFeatureDetector detector(2000);
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);


    }

    //different
    else if(strcmp(argv[3],"ORB")==0)
    {
        ORB detector(200);
        detector(image1,Mat(),keypoints1);
        detector(image2,Mat(),keypoints2);


    }
    /*
    else if(strcmp(argv[3],"BRISK")==0){
        BRISK detector;
        detector(image1,Mat(),keypoints1);
        detector(image2,Mat(),keypoints2);


    }
    */
    else if(strcmp(argv[3],"HARRIS")==0)
    {
        GoodFeaturesToTrackDetector detector( 1000,0.01,1., 3,true, 0.04);
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);



    }
    else if(strcmp(argv[3],"GFTT")==0)
    {
        GoodFeaturesToTrackDetector detector;
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);

    }

    else if(strcmp(argv[3],"DENSE")==0)
    {
        DenseFeatureDetector detector(0.5);
        detector.detect(image1, keypoints1);
        detector.detect(image2, keypoints2);

    }
    else if(strcmp(argv[3],"SIMPLE BLOB")==0)
    {
        cv::SimpleBlobDetector::Params params;
        params.minDistBetweenBlobs = 10.0;  // minimum 10 pixels between blobs
        params.filterByArea = true;         // filter my blobs by area of blob
        params.minArea = 10.0;              // min 20 pixels squared
        params.maxArea = 500.0;             // max 500 pixels squared
        SimpleBlobDetector detector(params);



        detector.detect(image1,keypoints1);
        detector.detect(image2,keypoints2);

    }
    else
    {

        cout << "\n detecteur inconnu"<<endl;
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
        cout <<"\n";
        exit(0);
    }


    //selection du descripteur



    if(strcmp(argv[4],"SIFT")==0)
    {
       
       //SiftDescriptorExtractor siftdesc;
       //*descriptor = siftdesc;
      //descriptor = new SiftDescriptorExtractor(10);
      //descriptor = DescriptorExtractor::create("SIFT");
       //*descriptor = new DescriptorExtractor("SIFT");
         //SiftDescriptorExtractor();
       //descriptor=descriptorAux;
        //descriptor = descriptorAux;
      descriptor = cv::DescriptorExtractor::create("ORB");


    }
    else
    {
        exit(0);
    }









    /*
    char* title1 = argv[3];
    char* title2;
    strcpy(title2,"transparence");
    strcat(title2,argv[3]);

    matches_window=title1;
    transparency_window=title2;
    */
    /*char* title1;
    title1 = "transparence";
    title1=title1 + argv[3];
    transparency_window=title1;
    */

    //transparency_window << "transparence " + argv[3];
    string st1=argv[3];
    string st2="transparence + ";

    string st3 = st2 + argv[3];

    transparency_window = st1;
    matches_window= st3;



    rows = image1.rows;
    cols = image1.cols;


    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);






    Mat descriptors1, descriptors2;

    (*descriptor).compute(image1, keypoints1, descriptors1);
    (*descriptor).compute(image2, keypoints2, descriptors2);



    // Construction of the matcher
    //BruteForceMatcher< HammingLUT > matcher;
    BruteForceMatcher < Hamming > matcher;	// =BruteForceMatcher<Hamming>(10);

    Mat descriptorAuxKp1;

    Mat descriptorVois;

    vector < int >associateIdx;
    vector < int >associateIdxVois;

    vector < KeyPoint > keypointsVois;


    //si on a trouvé un keypoint2 de l'image2 correspondant au keypoint1 de l'image 1 en cours
    bool aCorres = false;
    //nb de points reliés
    int pointCourant = 0;

    for (int i = 0; i < keypoints1.size(); i++)
    {
        //on a pas encore trouvé de correspondant pour kp1
        aCorres = false;
        //on copie la ligne i du descripteur, qui correspond aux différentes valeurs données par le descripteur pour le Keypoints[i]
        descriptors1.row(i).copyTo(descriptorAuxKp1);



//on supprime tous les keypointsvoisins associé au keypoint1 precedent precedants
        keypointsVois.erase(keypointsVois.begin(), keypointsVois.end());
        for (int j = 0; j < keypoints2.size(); j++)
        {

            float p1x = keypoints1[i].pt.x;
            float p1y = keypoints1[i].pt.y;
            float p2x = keypoints2[j].pt.x;
            float p2y = keypoints2[j].pt.y;

            float distance = sqrt( pow((p1x - p2x), 2) + pow( (p1y - p2y) ,  2));


            //parmis les valeurs dans descriptors2 on ne va garder que ceux dont les keypoints associés sont à une distance définie(e.g 4) du keypoints en cours(ici c'est le ieme keypoint).
            if (distance < 4)
            {

                //on a au moins un keypoint correspondant
                aCorres = true;
                KeyPoint kpVois;

                //on rajoute les pixels du 9 voisinage dans la liste des keypoints2 à matcher
                for (float ivois = -1; ivois < 2; ivois++)
                {

                    for (float jvois = -1; jvois < 2; jvois++)
                    {



                        kpVois.pt.x = p2x + ivois;
                        kpVois.pt.y = p2y + jvois;
                        kpVois.size = keypoints1[0].size;

                        //il faut que les coordonnes du point en question soient contenus dans l'image
                        if (kpVois.pt.x >= 0 && kpVois.pt.x < image1.rows
                                && kpVois.pt.y >= 0 && kpVois.pt.y < image1.cols)
                        {

                            keypointsVois.push_back(kpVois);

                        }
                    }
                }

            }			//fin de if(distance<4)




        }			//fin for(int j=0;j<descriptors2.rows;j++)




        //il faudra vérifier que le kp1 a un kp2 possible
        if (aCorres)
        {
            //on sait que le keypoints1[i] va être relié
            pointsx.push_back(keypoints1[i]);

            //on calcule le descripteur de tous les pixels retenus comme candidat

            (*descriptor).compute(image2, keypointsVois, descriptorVois);



            //ici on ne matche qu'un keypoints de l'image1 avec le meilleur des keypoints gardés de l'image 2
            matcher.match(descriptorAuxKp1, descriptorVois, matches);

            //on a trouvé le keypoints qui va le mieux
            //nrmlt on a trouvé un kp

            //cout << "\n On a trouvé " << matches.size() << " points correspondants";
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
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {

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


    for (int i = 0; i < matchesWithDist.size(); i++)
    {


        kp1x = pointsx[matchesWithDist[i].queryIdx].pt.x;
        kp1y = pointsx[matchesWithDist[i].queryIdx].pt.y;
        kp2x = pointsy[matchesWithDist[i].trainIdx].pt.x;
        kp2y = pointsy[matchesWithDist[i].trainIdx].pt.y;

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

