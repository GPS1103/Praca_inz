#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <glob.h>
#include <sys/types.h>
#include <dirent.h>
#include <cmath>

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier faceCascade;

string gstreamer() {
	int captureWidth = 3264 ;
	int captureHeight = 2464 ;
	int displayWidth =  640 ;
	int displayHeight = 480 ;
	int framerate = 21 ;
	int flipMethod = 2 ;

    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" +  to_string(captureWidth) + ", height=(int)" +
            to_string(captureHeight) + ", format=(string)NV12, framerate=(fraction)" +  to_string(framerate) +
           "/1 ! nvvidconv flip-method=" +  to_string(flipMethod) + " ! video/x-raw, width=(int)" +  to_string(displayWidth) + ", height=(int)" +
            to_string(displayHeight) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

int main(){
    string cascade = "/home/jetson/Testy/face_c++/haarcascade_frontalface_default.xml";
    //cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);
  
    //if (!capture.isOpened()) return 0;
    cv::Ptr<cv::cuda::CascadeClassifier> faceCascadeGpu = cv::cuda::CascadeClassifier::create(cascade);
    CascadeClassifier faceCascade;
    faceCascade.load(cascade);

    faceCascadeGpu->setFindLargestObject(false);
    faceCascadeGpu->setScaleFactor(1.05);
    faceCascadeGpu->setMinNeighbors(5);
	faceCascadeGpu->setMinObjectSize(Size(80,80));
	faceCascadeGpu->setMaxObjectSize(Size(150,150));
    
    TickMeter timer;
    // if (!faceCascade.load("/home/jetson/Testy/face_c++/haarcascade_frontalface_default.xml"))
    // faceCascade.detectMultiScale(frameEq, faces, 1.05, 5, 0, Size(80, 80), Size(180,180));
    Mat frame, gray, result;
    cv::cuda::GpuMat frameGpu, grayGpu, resultGpu, facesBuf;
    vector<Rect> faces;
    int iter = 0; 
    float cvtCPU = 0, cvtGPU = 0, resizeCPU = 0, resizeGPU = 0, detectCPU = 0, detectGPU; 
    frame = imread("/home/jetson/Testy/face_c++/hyundai.jpg");
    while(iter != 3){
        
        //capture >> frame;
        frameGpu.upload(frame);
        timer.start();
        cv::cvtColor(frame, gray, COLOR_BGR2GRAY);
        timer.stop();
        cout << "\nCPU cvtColor: " + to_string(timer.getTimeMilli()) +" ms"<< endl;
        cvtCPU += timer.getTimeMilli();
        
        timer.reset();
        timer.start();
        cv::cuda::cvtColor(frameGpu, grayGpu, COLOR_BGR2GRAY);
        timer.stop();
        cvtGPU += timer.getTimeMilli();
        cout << "GPU cvtColor: " + to_string(timer.getTimeMilli()) +" ms"<< endl;
        
        
        timer.reset();
        timer.start();
        cv::resize(gray, result, Size(200, 200), 0, 0, INTER_LINEAR);
        timer.stop();
        resizeCPU += timer.getTimeMilli();
        cout << "CPU resize: " + to_string(timer.getTimeMilli()) +" ms"<< endl;
        
        timer.reset();
        timer.start();
        cv::cuda::resize(grayGpu, resultGpu, Size(200, 200), 0, 0, INTER_LINEAR);
        timer.stop();
        resizeGPU += timer.getTimeMilli();
        cout << "GPU resize: " + to_string(timer.getTimeMilli()) +" ms"<< endl;

        timer.reset();
        timer.start();
        faceCascade.detectMultiScale(gray, faces, 1.05, 5, 0, Size(80, 80), Size(150,150));
        timer.stop();
        detectCPU += timer.getTimeMilli();
        cout << "CPU detect: " + to_string(timer.getTimeMilli()) +" ms"<< endl;
        
        timer.reset();
        timer.start();
        faceCascadeGpu->detectMultiScale(grayGpu, facesBuf);
        timer.stop();
        detectGPU += timer.getTimeMilli();
        cout << "GPU detect: " + to_string(timer.getTimeMilli()) +" ms"<< endl;

        iter++;
        }
        cout << "\nWartości średnie:" << endl;
        cout << "CPU cvtColor: " + to_string(cvtCPU/iter) +" ms"<< endl;
        cout << "GPU cvtColor: " +to_string(cvtGPU/iter) +" ms"<< endl;

        cout << "CPU resize: " + to_string(resizeCPU/iter) +" ms"<< endl;
        cout << "GPU resize: " +to_string(resizeGPU/iter) +" ms"<< endl;

        cout << "CPU detectMultiScale: " + to_string(detectCPU/iter) +" ms" << endl;
        cout << "GPU detectMultiScale: " +to_string(detectGPU/iter) +" ms" << endl;
    return 0;
}