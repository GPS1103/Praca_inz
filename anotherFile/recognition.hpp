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

using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier faceCascade;
string path = "/home/jetson/Testy/face_c++/photos/";
string eigenClassifier = "/home/jetson/Testy/face_c++/Classifier/eigenface.xml";
string name;
bool showInfo = false;
int imgCounter = 0;
int imgHeight = 128; 
int imgWidth = 64;

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

void showAdvancedInfo(Mat frame, float fps){
	ostringstream ss;
    ss << "FPS: " << setprecision(1) << fixed << fps;
	putText(frame, ss.str(), Point(2, 20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 0.8, 16);
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
	Mat frameGray, frameEq;
	Mat frameCrop = frame.clone();
	Mat croppedImage, resultOfCropped, grayCropped;
	
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);
	equalizeHist(frameGray, frameEq);

	TickMeter timer;
    timer.start();
	
	faceCascade.detectMultiScale(frameEq, faces, 1.05, 5, 0, Size(80, 80), Size(150,150));

	for (size_t iter = 0; iter < faces.size(); iter++)

	{	
		if(makePhoto){

			croppedImage = frameCrop(faces[0]);
			resize(croppedImage, resultOfCropped, Size(imgWidth, imgHeight), 0, 0, INTER_LINEAR);
			cvtColor(resultOfCropped, grayCropped, COLOR_BGR2GRAY);
			string folderPath = path + name + "/";

			stringstream nameStream;
			nameStream << folderPath.c_str() << imgCounter << ".jpg";
			string imageName = nameStream.str();
			imwrite(imageName, grayCropped);
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
	cin >> name;

	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);
  
    if (!capture.isOpened())  
        return;

    if (!faceCascade.load("/home/jetson/Testy/face_c++/haarcascade_frontalface_default.xml"))
    {
        cout << "Nie wgrano klasyfikatora" << endl;
        return ;
    };

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

vector<cv::String> split(const std::string &txt, std::vector<std::string> &strs)
{
    size_t pos = txt.find(' ');
    size_t initialPos = 0;
    strs.clear();

    while( pos != std::string::npos ) {
        strs.push_back( txt.substr( initialPos, pos - initialPos ) );
        initialPos = pos + 1;

        pos = txt.find(' ', initialPos );
    }

    strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

    return strs;
}

void faceDetect() {

	//Scalar fontColor = Scalar(255);
	Scalar fontColor = CV_RGB(0, 255, 0);
	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();

	model->read(eigenClassifier);

	if (!faceCascade.load("/home/jetson/Testy/face_c++/haarcascade_frontalface_default.xml")) {
		cout << "Błąd podczas ładowania klasyfikatora" << endl;
		return;
	}

	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);

	if (!capture.isOpened())
	{
		cout << "Błąd: Nie utworzono sesji Gstreamer" << endl;
		return;
	}
	
	string windowName = "Rozpoznawanie twarzy";
	namedWindow(windowName, 1);

	while (true)
	{
		vector<Rect> faces;
		Mat frame, frameCopy, grayFrame, eqFrame;
		TickMeter timer;

		capture >> frame;
		
		if (!frame.empty()) {

    		timer.start();
			frameCopy = frame.clone();
			cvtColor(frameCopy, grayFrame, COLOR_BGR2GRAY);
			equalizeHist(grayFrame, eqFrame);
			faceCascade.detectMultiScale(eqFrame, faces, 1.05, 5, 0, Size(80, 80), Size(180, 180));

			for (int i = 0; i < faces.size(); i++)
			{
				Mat face = eqFrame(faces[i]);
				Mat faceResized;
				cv::resize(face, faceResized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_LINEAR);

				int label = -1;
				double confidence = 0.0;
				model->predict(faceResized, label, confidence);

				cout << "confidence " << confidence << " Label: " << label << endl;
				std::vector<std::string> vectorLabel;
				if(model->getLabelInfo(label) == "Adi") { split( "Adrian Sobiesierski", vectorLabel);  }
				else vectorLabel.push_back(model->getLabelInfo(label));

				if(confidence < 5000){
					rectangle(frame, faces[i], fontColor, 2);

					for(size_t s = 0; s <vectorLabel.size(); s++ ){
						rectangle(frame, Rect(faces[i].x -1, faces[i].y + faces[i].height + 1 + (faces[i].height * 0.2 * s), faces[i].width +2, faces[i].height * 0.2), fontColor, -1);
						putText(frame, vectorLabel[s], Point(faces[i].x + 1, faces[i].y + faces[i].height * 1.13 + (faces[i].height * 0.2 * s)), FONT_HERSHEY_SIMPLEX, (float)faces[i].height/190, Scalar(255, 255, 255), 0.5, 1);				
					}

				}
			}
			//putText(frameCopy, "No. of Persons detected: " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
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
	