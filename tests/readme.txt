
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

SimpleBlobDetector(const SimpleBlobDetector::Params &parameters = SimpleBlobDetector::Params());

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


//BruteForce

//BruteForce-L1

//BruteForce-Hamming

//FlannBased



---------------------------------------------------------------

comment utiliser 2string:


    string ty = type2str(matrix.type());
    printf("Matrix: %s %dx%d n", ty.c_str(), matrix.cols, matrix.rows);

