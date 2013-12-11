#include <opencv2/opencv.hpp>

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
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



/*rayon autour duquel on cherche un kp2 autour d'un kp1*/
int rayonDist = 2;
/*rayon du cercle ou du carré dans lequel on va regarder tous les pixels quand
 *on a trouvé un kp2 correspondant à un kp1
 */


int rayonVois = 1;




vector < KeyPoint > keypoints1, keypoints2, pointsx, pointsy;
vector < DMatch > matches;

vector < DMatch > matchesWithDist;


/// Function header
void interface(int argc, void *);

/**
 * @function main
 */



int main(int argc, char **argv)
{

  
   Ptr<DescriptorExtractor> descriptor;  
   Ptr<DescriptorMatcher> matcher;
   
   
   
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

        cout << "\n\n pour les descripteurs SIFT et SURF\n";
        cout << "\n BruteForce";
        cout << "\n BruteForce-L1";

         cout << "\n\n pour les descripteurs BRIEF et ORB\n";
        cout << "\n BruteForce-Hamming";
        cout << "\n FlannBased";
        cout <<"\n";
        cout <<endl;
        

        exit(0);
    }



    image1 = imread(argv[1], 1);
    image2 = imread(argv[2], 1);
    


    //selection du detecteur

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

    //maniere differente d'implementer
    else if(strcmp(argv[3],"ORB")==0)
    {
        ORB detector(200);
        detector(image1,Mat(),keypoints1);
        detector(image2,Mat(),keypoints2);


    }
    
    
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
        Ptr<FeatureDetector> detector;
        detector = FeatureDetector::create("Dense");
        
        //DenseFeatureDetector detector(0.5);
        detector->detect(image1, keypoints1);
        detector->detect(image2, keypoints2);

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



    if(strcmp(argv[4],"BRIEF")==0)
    {
       
        descriptor = cv::DescriptorExtractor::create("BRIEF");

    }
    else if(strcmp(argv[4],"ORB")==0)
    {
       descriptor = cv::DescriptorExtractor::create("ORB");

    }
     else if(strcmp(argv[4],"SIFT")==0)
    {
      
      descriptor = cv::DescriptorExtractor::create("SIFT");

    }
    else if(strcmp(argv[4],"SURF")==0){

       descriptor = cv::DescriptorExtractor::create("SURF");
      
    }




    else
    {

        cout << "\n descripteur inconnu"<<endl;
        cout << "\n liste des descripteurs possible:";
        cout << "\n SIFT";
        cout << "\n SURF";
        cout << "\n BRIEF";
        cout << "\n ORB";
        cout <<"\n";
 
        exit(0);
    }

    if(strcmp(argv[5],"BruteForce")==0 || strcmp(argv[5],"BruteForce-L1")==0 || strcmp(argv[5],"BruteForce-Hamming")==0 || strcmp(argv[5],"FlannBased")==0){
    matcher = cv::DescriptorMatcher::create(argv[5]);


     
    }
 
    else{
        cout << "\n apparieur inconnu"<<endl;
        cout << "\n liste des apparieurs possible:";
        cout << "\n BruteForce";
        cout << "\n BruteForce-L1";
        cout << "\n BruteForce-Hamming";
        cout << "\n FlannBased";
        cout <<"\n";
 
        exit(0);
     

    }

   
    

    string st1= argv[3] + (string)" + " +argv[4] +  (string)" + " + argv[5];
    string st2="transparence " + st1;

    transparency_window = st2;
    matches_window= st1;



    rows = image1.rows;
    cols = image1.cols;


    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    namedWindow("image2", WINDOW_AUTOSIZE);
    imshow("image2", image2);






    Mat descriptors1, descriptors2;
    
        (*descriptor).compute(image1, keypoints1, descriptors1);
        (*descriptor).compute(image2, keypoints2, descriptors2);

    

    Mat descriptorAuxKp1;
    descriptorAuxKp1.create(0,0,CV_8UC1);
    Mat descriptorVois;
    descriptorVois.create(0,0,CV_8UC1);
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
            if (distance < rayonDist)
            {

                //on a au moins un keypoint correspondant
                aCorres = true;
                KeyPoint kpVois;

                //on rajoute les pixels du 9 voisinage dans la liste des keypoints2 à matcher
                for (float ivois = -rayonVois; ivois < rayonVois+1; ivois++)
                {

                    for (float jvois = -rayonVois; jvois < rayonVois+1; jvois++)
                    {



                        kpVois.pt.x = p2x + ivois;
                        kpVois.pt.y = p2y + jvois;
                        kpVois.size = keypoints1[0].size;

                        //il faut que les coordonnes du point en question soient contenus dans l'image


                        if (kpVois.pt.x >= 0 && kpVois.pt.x < image1.rows
                                && kpVois.pt.y >= 0 && kpVois.pt.y < image1.cols)
                        {

                            
                           //si on veut décrire un cercle
                           if(sqrt(pow(ivois,2)+pow(jvois,2)) < rayonVois){

                                    keypointsVois.push_back(kpVois);
                                   /* image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[0] = 0;
                                    image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[1] = 0;
                                    image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[2] = 0;*/


                           }
			   //ou un carre
                           /*image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[0] = 0;
                                    image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[1] = 0;
                                    image1.at < cv::Vec3b > (kpVois.pt.y,kpVois.pt.x)[2] = 0;
                           keypointsVois.push_back(kpVois);*/
                           


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
   

             matcher->match(descriptorAuxKp1, descriptorVois, matches);  

            //on a trouvé le keypoints qui va le mieux
            //nrmlt on a trouvé un kp

            
            KeyPoint best2 = keypointsVois[matches[0].trainIdx];

            pointsy.push_back(best2);

            matches[0].trainIdx = pointCourant;
            matches[0].queryIdx = pointCourant;



            pointCourant++;
            matchesWithDist.insert(matchesWithDist.end(), matches.begin(), matches.end());


        }




    }	//fin du for i



//ici on trie les matchesWithDist par distance des valeurs des descripteurs et non par distance euclidienne
    nth_element(matchesWithDist.begin(), matchesWithDist.begin() + 24, matchesWithDist.end());
    // initial position
    // position of the sorted element
    // end position

    //si on veut garder les meilleurs correspondances

     if(matchesWithDist.size()>500){
       matchesWithDist.erase(matchesWithDist.begin() + 500,matchesWithDist.end());
     }

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
