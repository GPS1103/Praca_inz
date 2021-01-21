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

string cascade = "./Classifier/haarcascade_frontalface_default.xml";
cv::Ptr<cv::cuda::CascadeClassifier> faceCascade = cv::cuda::CascadeClassifier::create(cascade);
string path = "./photos/";
string eigenClassifier = "./Classifier/eigenface.xml";
string name;
bool showInfo = false;
int imgCounter = 0;
int imgHeight = 128; 
int imgWidth = 64;

string gstreamer() {
	int captureWidth = 3264 ;
	int captureHeight = 2464;
	int displayWidth =  1024 ;
	int displayHeight = 768 ;
	int framerate = 21 ;
	int flipMethod = 2 ;

    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" +  to_string(captureWidth) + ", height=(int)" +
            to_string(captureHeight) + ", format=(string)NV12, framerate=(fraction)" +  to_string(framerate) +
           "/1 ! nvvidconv flip-method=" +  to_string(flipMethod) + " ! video/x-raw, width=(int)" +  to_string(displayWidth) + ", height=(int)" +
            to_string(displayHeight) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

void showAdvancedInfo(Mat frame, float fps){
	ostringstream ss;
    ss << "FPS: " << setprecision(1) << fixed << fps;
	putText(frame, ss.str(), Point(2, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 0.8, 16);
}

void initialize(){
	faceCascade->setFindLargestObject(false);
    faceCascade->setScaleFactor(1.05);
    faceCascade->setMinNeighbors(5);
	faceCascade->setMinObjectSize(Size(80,80));
	faceCascade->setMaxObjectSize(Size(150,150));
}
vector<cv::String> listOfDirectories(){
	vector<cv::String> directories;
	DIR* dirp = opendir(path.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {

		if(!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..") ) ;
		else directories.push_back(dp->d_name);
    }
    closedir(dirp);

	return directories;
}

void makeFolder(){
	string folderPath = path + name + "/";

	if (mkdir(folderPath.c_str(), 0777) == -1) 
        cerr << "Błąd: " << strerror(errno) << endl; 
	else
		cout << "Utworzono folder: " + folderPath;
}


void showInfoWhenMakeSelfies(Mat frame){
	//ostringstream ss;
    //ss << "FPS: " << setprecision(1) << fixed << fps;
	putText(frame, "Zostanie zrobione 20 zdjec.", Point(2, 10), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 0.8, 16);
	putText(frame, "Upewnij sie, ze jestes sam w kadrze.", Point(2, 22), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 0.8, 16);
	putText(frame, "Patrz w obiektyw.", Point(2, 32), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 0.8, 16);

	if(imgCounter != 0)
	{
		putText(frame, "Zrobiono " + to_string(imgCounter+1) + " zdjec", Point(2, frame.rows - 5), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 0.8, 16);
	}
}

void detectAndDisplay(Mat frame, bool makePhoto)
{
	vector<Rect> faces;
	cv::cuda::GpuMat frameGray, frameEq, frameCropGpu;
	cv::cuda::GpuMat frameGpu, croppedImage, resultOfCropped, grayCropped, facesBuf;
	Mat frameCrop;
	frameGpu.upload(frame);
	cv::cuda::cvtColor(frameGpu, frameGray, COLOR_BGR2GRAY);
	frameCropGpu = frameGray.clone();
	//cv::cuda::equalizeHist(frameGray, frameEq);
	
	TickMeter timer;
    timer.start();
	
	faceCascade->detectMultiScale(frameGray, facesBuf);
    faceCascade->convert(facesBuf, faces);

	cout << "face: " + to_string(faces.size()) << endl;

	for (size_t iter = 0; iter < faces.size(); iter++)
	{	
		if(makePhoto){

			croppedImage = frameCropGpu(faces[0]);
			cv::cuda::resize(croppedImage, resultOfCropped, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
			string folderPath = path + name + "/";

			stringstream nameStream;
			nameStream << folderPath.c_str() << imgCounter << ".jpg";
			string imageName = nameStream.str();
			resultOfCropped.download(frameCrop);
			imwrite(imageName, frameCrop);
			imgCounter++;

			cout << "Zrobiono zdjęcie: " + imageName << endl;
		}

		Point pt1(faces[iter].x, faces[iter].y);
		Point pt2((faces[iter].x + faces[iter].height), (faces[iter].y + faces[iter].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
		
	}
	timer.stop();
	if(showInfo) showAdvancedInfo(frame, timer.getFPS());

	imshow("Sesja zdjeciowa", frame);

}

void makeAndSaveImages()
{   

    imgCounter = 0;
	cout << "\nWprowadz swoje imie:  ";
	std::getline(std::cin >> std::ws, name);

	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);
  
    if (!capture.isOpened()) return;

	initialize();
    Mat frame;
	cout << "\nSesja zdjeciowa" << endl;
	char key;
	int i = 0;
	makeFolder();
	TickMeter timer;
    timer.start();

	while(true)
    {
        capture >> frame;

		if (imgCounter == 20)
		{
			cout << "Zakończono sesje zdjęciową" << endl;
			break;
		}
	
        char key = waitKey(5);

        if (key == 27) break;
		else if (key == 'i'){
			showInfo = !showInfo;
		}
	
		cout << to_string(timer.getTimeMilli()) << endl;
		showInfoWhenMakeSelfies(frame);
		timer.stop();

		if(timer.getTimeMilli() >= 500) 
		{
			detectAndDisplay(frame, true);
			timer.reset();
		}
		else 
		{	
			timer.start();
			detectAndDisplay(frame, false);
		}
	}
	
    return;
}

void getImages(vector<Mat>& images, vector<int>& labels, Ptr<EigenFaceRecognizer>& model) {

	vector<cv::String> directories = listOfDirectories();
	vector<cv::String> photoDir;

	size_t countDirectories = directories.size();
	for(size_t dir = 0; dir < countDirectories; dir++){

		glob(path + directories[dir] + "/", photoDir, true);
		size_t countPhotos = photoDir.size();

		cv::String labelName = directories[dir] ;
		model->setLabelInfo(dir, labelName);

		for (size_t photo = 0; photo < countPhotos; photo++)
		{	
			images.push_back(imread(photoDir[photo], 0));
			labels.push_back(dir);
		}
	}	
}

void eigenClassifierTraining() {
	vector<Mat> images;
	vector<int> labels;

	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
	getImages(images, labels, model);

	cout << "Liczba zdjęć: " << images.size() << endl;
	
	model->train(images, labels);
	model->save(eigenClassifier);

	cout << "Zakończono trening" << endl;  

}

void faceDetect() {

	//Scalar fontColor = Scalar(255);
	Scalar fontColor = CV_RGB(0, 255, 0);
	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();
	model->read(eigenClassifier);

	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);

	if (!capture.isOpened())
	{
		cout << "Błąd: Nie utworzono sesji Gstreamer" << endl;
		return;
	}

	initialize();
	string windowName = "Rozpoznawanie twarzy";
	namedWindow(windowName, 1);

	Mat frame;
	Mat facePredict;

	cv::cuda::GpuMat frameGpu, grayFrame, eqFrame, faceResized, facesBuf;

	while (true)
	{
		vector<Rect> faces;

		TickMeter timer;

		capture >> frame;
		
		if (!frame.empty()) {

    		timer.start();
			frameGpu.upload(frame);

			cv::cuda::cvtColor(frameGpu, grayFrame, COLOR_BGR2GRAY);
	
			faceCascade->detectMultiScale(grayFrame, facesBuf);
			faceCascade->convert(facesBuf, faces);

			for (int i = 0; i < faces.size(); i++)
			{
				cv::cuda::GpuMat face = grayFrame(faces[i]);
				cv::cuda::resize(face, faceResized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_LINEAR);
				faceResized.download(facePredict);
				int label = -1;
				double confidence = 0;
				
				model->predict(facePredict, label, confidence);

				cout << "confidence " << confidence << " Label: " << label << endl;
				if(confidence < 7000){
					rectangle(frame, faces[i], fontColor, 2);
					rectangle(frame, Rect(faces[i].x -1, faces[i].y + faces[i].height + 1, faces[i].width +2, faces[i].height * 0.2), fontColor, -1);
					putText(frame, model->getLabelInfo(label), Point(faces[i].x + 2, faces[i].y + faces[i].height * 1.2 - 1), FONT_HERSHEY_SIMPLEX, (float)faces[i].height/115, Scalar(255, 255, 255), 0.8, 16);
				}
			}

			timer.stop();
			if(showInfo) showAdvancedInfo(frame, timer.getFPS());
			imshow(windowName, frame);
		}
		char key = waitKey(5);

		if (key == 27) break;
		else if (key == 'i'){
			showInfo = !showInfo;
		}
	}
}
	