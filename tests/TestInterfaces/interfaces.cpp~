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



void callbackButton(int argc, void *);




int main(int argc, char *argv[]){
	cout <<"ahaha";
    int value = 50;
    int value2 = 2;

	void* pointer;

    cvNamedWindow("main1",CV_WINDOW_NORMAL);
    //cvNamedWindow("main2",CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL);

    cvCreateTrackbar( "track1", "main1", &value, 255,  NULL);//OK tested
    //string nameb1 = "button1";
    //string nameb2 = "button2";
    //cvCreateButton("button1",callbackButton,NULL,CV_CHECKBOX,1);
    //cvCreateButton("button1",callbackButton,NULL,CV_CHECKBOX,1);
    //  char* nameb2 = "button2";
    //const char* b = "button2";
    cvCreateButton(NULL,callbackButton);
    /*cvCreateTrackbar( "track2", NULL, &value2, 255, NULL);
    cvCreateButton("button5",callbackButton1,NULL,CV_RADIOBOX,0);
    cvCreateButton("button6",callbackButton2,NULL,CV_RADIOBOX,1);

   // cvSetMouseCallback( "main2",on_mouse,NULL );
    */

    IplImage* img1 = cvLoadImage(argv[1]);
    //IplImage* img2 = cvCreateImage(cvGetSize(img1),8,3);
    /*CvCapture* video = cvCaptureFromFile(argv[2]);
    IplImage* img3 = cvCreateImage(cvGetSize(cvQueryFrame(video)),8,3);
	*/
    while(cvWaitKey(33)%256 != 27)
    {
	/*
        cvAddS(img1,cvScalarAll(value),img2);
        cvAddS(cvQueryFrame(video),cvScalarAll(value2),img3);
	*/	
        cvShowImage("main1",img1);
        //cvShowImage("main2",img3);
    }

    cvDestroyAllWindows();
   /* cvReleaseImage(&img1);
    cvReleaseImage(&img2);
    cvReleaseImage(&img3);
    cvReleaseCapture(&video);*/
	
    return 0;
}

void callbackButton(int argc, void *){
}




