#include <stdlib.h>
#include <unistd.h>
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
#include <map>
#include <string>
#include <jsoncpp/json/json.h>
#include <curl/curl.h>
#include <Python.h>

#include <opencv2/photo.hpp>
#include <opencv2/ml.hpp>
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
using namespace cv::ml;

string cascade = "Classifier/haarcascade_frontalface_default.xml";
cv::Ptr<cv::cuda::CascadeClassifier> faceCascade = cv::cuda::CascadeClassifier::create(cascade);
string positiveDirectory = "photos/positive/";
string negDirectory = "photos/negative";
string svmClassifier = "Classifier/svm.xml";
string labelsFile = "labels.json";
string name;

bool showInfo = false;
int imgCounter = 0;
int imgWidth = 64;
int imgHeight = 128; 
bool test = false;

string gstreamer() {
	int captureWidth = 3264 ;
	int captureHeight = 2464;
	int displayWidth =  640 ;
	int displayHeight = 480 ;
	int framerate = 21 ;
	int flipMethod = 2 ;

    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" +  to_string(captureWidth) + ", height=(int)" +
            to_string(captureHeight) + ", format=(string)NV12, framerate=(fraction)" +  to_string(framerate) +
           "/1 ! nvvidconv flip-method=" +  to_string(flipMethod) + " ! video/x-raw, width=(int)" +  to_string(displayWidth) + ", height=(int)" +
            to_string(displayHeight) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
void initializeHAAR(){
	faceCascade->setFindLargestObject(false);
    faceCascade->setScaleFactor(1.05);
    faceCascade->setMinNeighbors(6);
	faceCascade->setMinObjectSize(Size(80,80));
	faceCascade->setMaxObjectSize(Size(150,150));
}

vector<cv::String> listOfDirectories(){
	vector<cv::String> directories;
	DIR* dirp = opendir(positiveDirectory.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {

		if(!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..") ) ;
		else directories.push_back(dp->d_name);
    }
    closedir(dirp);

	return directories;
}

void makeFolder(){
	string folderPath = positiveDirectory + name + "/";

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
			string folderPath = positiveDirectory + name + "/";

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
	cout << "\nWprowadź swoje imię i nazwisko:  ";
	std::getline(std::cin >> std::ws, name);
	cout << name << endl;
	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);
  
    if (!capture.isOpened()) return;

	initializeHAAR();
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

		if (imgCounter == 50)
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

void getImages(vector<Mat>& images, vector<int>& labels) {

	vector<cv::String> directories = listOfDirectories();
	vector<cv::String> photoDir;
	Json::Value event;   

	size_t countDirectories = directories.size();
	for(size_t dir = 0; dir < countDirectories; dir++){

		glob(positiveDirectory + directories[dir] + "/", photoDir, true);
		size_t countPhotos = photoDir.size();

		Mat imgNoising, imgDenoising;
		float mean = (10,13,18);
		float sigma = (1,5,15);
			
		clog << "Dir name: " << directories[dir] << " Label: " << dir << endl;
		for (size_t photo = 0; photo < countPhotos; photo++)
		{	
			Mat img = imread(photoDir[photo]);
			//Mat noise = Mat(img.size(), img.type());
			//imgNoising = img.clone();

			images.push_back(img);
			labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/img.jpg", img);
			
			// GaussianBlur(img, imgDenoising, Size(9, 9), 0);
			// images.push_back(imgDenoising);
			// labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/denoise.jpg", imgDenoising);

			// cv::randn(noise, mean, sigma);
			// imgNoising+= noise;
			// images.push_back(imgNoising);
			// labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/noise.jpg", imgNoising);
	
		    flip(img, img, 1);
			images.push_back(img);
			labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/flip.jpg", img);

			// flip(imgDenoising, imgDenoising, 1);
			// imgDenoising.convertTo(imgDenoising, -1, 1, 20);
			// images.push_back(imgDenoising);
			// labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/flipDenoise.jpg", imgDenoising);
				
			// flip(imgNoising, imgNoising, 1);
			// images.push_back(imgNoising);
			// labels.push_back(dir);
			// if(test) imwrite("/home/jetson/Testy/test/flipNoise.jpg", imgNoising);
				
			// test = false;
		}

		event["labels"][std::to_string(dir)] = directories[dir];
    
   // cout << styledWriter.write(event);
	}	

	Json::StyledWriter styledWriter;
	ofstream myfile;
    myfile.open (labelsFile);
    myfile << styledWriter.write(event);
    myfile.close();
}

void getNegativeImages(vector<Mat>& images, vector<int>& labels) {

    vector<cv::String> photoDir;
	glob(negDirectory, photoDir, true);
	size_t countPhotos = photoDir.size();

    Mat imgNoising, imgDenoising;
    float mean = (10,15,20);
    float sigma = (1,5,25);

	for (size_t photo = 0; photo < countPhotos; photo++)
	{	
		Mat img = imread(photoDir[photo]);
       // Mat noise = Mat(img.size(), img.type());
       // imgNoising = img.clone();

		images.push_back(img);
		labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/img.jpg", img);
           
        // GaussianBlur(img, imgDenoising, Size(9, 9), 0);
        // images.push_back(imgDenoising);
		// labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/denoise.jpg", imgDenoising);

        // cv::randn(noise, mean, sigma);
        // imgNoising+= noise;
        // images.push_back(imgNoising);
		// labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/noise.jpg", imgNoising);
  
        // flip(img, img, 1);
        // images.push_back(img);
		// labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/flip.jpg", img);

        // flip(imgDenoising, imgDenoising, 1);
        // imgDenoising.convertTo(imgDenoising, -1, 1, 20);
        // images.push_back(imgDenoising);
		// labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/flipDenoise.jpg", imgDenoising);
            
        // flip(imgNoising, imgNoising, 1);
        // images.push_back(imgNoising);
		// labels.push_back(-1);
        // if(test) imwrite("/home/jetson/Testy/test/flip.jpg", imgNoising);
            
        //test = false;
		}

}

void convertToMl( const vector< Mat > & trainSamples, Mat& trainData )
{
    const int rows = (int)trainSamples.size();
    const int cols = (int)std::max( trainSamples[0].cols, trainSamples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); 
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < trainSamples.size(); ++i )
    {	
		CV_Assert( trainSamples[i].cols == 1 || trainSamples[i].rows == 1 );

        if( trainSamples[i].cols == 1 )
        {	
            transpose( trainSamples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( trainSamples[i].rows == 1 )
        {	
            trainSamples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void computeHOGs( const Size wsize, const vector< Mat > & imagesVector, vector< Mat > & gradientsList )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;

    for( size_t i = 0 ; i < imagesVector.size(); i++ )
    {
        if ( imagesVector[i].cols >= wsize.width && imagesVector[i].rows >= wsize.height )
        {
            hog.compute( imagesVector[i], descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradientsList.push_back( Mat( descriptors ).clone() );            
        }
    }
}

void trainSVM(){

    vector< Mat > imagesVector, gradientsList;
    vector< int > labels;

    clog << "Inicjalizacja pozytywnych zdjęć..." << endl ;
    getImages(imagesVector, labels);

    int posImages = imagesVector.size();
    CV_Assert(posImages != 0);
    clog << "...załadowano " << posImages << " zdjęć." << endl;

    Size imagesVectorSize = imagesVector[0].size();
    clog << "Rozmiar zdjęć: " << imagesVectorSize << endl;

    clog << "Inicjalizacja negatywnych zdjęć...";
	getNegativeImages(imagesVector, labels);
    int negImages = imagesVector.size() - posImages;
    CV_Assert(posImages != imagesVector.size());
    clog << "...załadowano " << negImages << " zdjęć." << endl;

    clog << "Obliczanie HOG dla wszystkich zdjęć...";
    computeHOGs( imagesVectorSize, imagesVector, gradientsList);

    clog << "...zakończono ( Liczba zdjęć: " << imagesVector.size() << " Etykieta: " <<  labels.size() << " )" << endl;
	clog << "Gradient: " << gradientsList[0].size() << endl;
    Mat trainData;
    convertToMl( gradientsList, trainData );

    clog << "Trenowanie SVM..." << endl << "wiersze: " << trainData.rows << endl << "kolumny: " << trainData.cols << endl;
    Ptr<SVM> svm = SVM::create();
    Ptr<ParamGrid> CvParamGrid_C = cv::ml::ParamGrid::create(pow(2.0, -5), pow(2.0, 15), pow(2.0, 2));
	Ptr<ParamGrid> CvParamGrid_gamma = cv::ml::ParamGrid::create(pow(2.0, -15), pow(2.0, 3), pow(2.0, 2));
	//Ptr<ParamGrid> CvParamGrid_C = cv::ml::ParamGrid::create(pow(2.0, 1), pow(2.0, 4), pow(2.0, 2));
	//Ptr<ParamGrid> CvParamGrid_gamma = cv::ml::ParamGrid::create(pow(2.0, 3), pow(2.0, 6), pow(2.0, 2));
    svm->setDegree( .65 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER, 1e3, 1e-6) );
    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::POLY);
	svm->setC(20);
	svm->setGamma(100);
	//svm->setDegree(.65);
    //svm->setGamma(10);    
	//svm->setCoef0( 1 );
    //svm->setDegree( 3 );
     //svm->setTermCriteria(	 TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    //svm->setGamma( 20 );
    // svm->setKernel( SVM::POLY );
    //svm->setNu( 0.5 );
    //svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    //svm->setC( 6 ); // From paper, soft classifier
    //svm->setType( SVM::C_SVC ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	
    //convertToMl( gradientsList, trainData );
	//svm->train( trainData, ROW_SAMPLE, labels );
	svm->trainAuto( trainData, ROW_SAMPLE, labels);//,  2, CvParamGrid_C, CvParamGrid_gamma );
	clog << "...zakończono " << endl;

    if (false)
    {
        gradientsList.clear();
		clog << "Ponowne obliczanie HOG dla wszystkich zdjęć...";
        computeHOGs( imagesVectorSize, imagesVector, gradientsList);
        int positive_count = gradientsList.size();
        clog << "...zakończono ( liczba zdjęć : " << imagesVector.size() << " )" << endl;

        clog << "Ponowne trenowanie SVM...";
        convertToMl( gradientsList, trainData );
        svm->train( trainData, ROW_SAMPLE, labels );
        clog << "...zakończono" << endl;
    }

	svm->save(svmClassifier);
}

void sendImagesToWeb(){
  
	CURL *curl = curl_easy_init();
	curl_mime *mime;
	curl_mimepart *part;

	string url = "http://10.42.0.36:8000/api/frame"; 

	mime = curl_mime_init(curl);
	part = curl_mime_addpart(mime);
	
	struct curl_slist *headers = NULL;
	static const char buf[] = "Expect:";
	headers = curl_slist_append(headers, buf);
	
	curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

	//`	curl_easy_perform(curl);
	curl_easy_cleanup(curl);
	curl_mime_free(mime);
	
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


int calculateFrequency(vector<int> &detectionVector){

	int detectionSize = detectionVector.size();

	map<int,unsigned> frequencyCount;
	for(int i = 0; i < detectionSize; i++){
		frequencyCount[detectionVector[i]]++;
	}

	unsigned currentMax = 0;
	int predictedPerson = 0;

	for(auto it = frequencyCount.cbegin(); it != frequencyCount.cend(); ++it ){
		if (it ->second > currentMax) {
			predictedPerson = it->first;
			currentMax = it->second;
		}
	}
	cout << "Wartość: " << predictedPerson << " Liczba wystąpień: " << currentMax << endl;
	if ( currentMax > detectionSize*0.9){
		return predictedPerson;
	}
	else return -1;
}

void faceDetect() {

	Scalar fontColor = CV_RGB(0, 255, 0);
	Ptr< cv::ml::SVM > svm = Algorithm::load<cv::ml::SVM>(svmClassifier);//cv::ml::SVM::create();

	Mat SV = svm->getSupportVectors();
	Mat USV = svm->getUncompressedSupportVectors();

	cout << "Support Vectors: " << SV.rows << endl;
	cout << "Uncompressed Support Vectors: " << USV.rows << endl;

	cv::VideoCapture capture(gstreamer(), cv::CAP_GSTREAMER);

	if (!capture.isOpened())
	{
		cout << "Błąd: Nie utworzono sesji Gstreamer" << endl;
		return;
	}

	initializeHAAR();
	string windowName = "Multimodalny system identyfikacji biometrycznej";
	namedWindow(windowName, 1);

	ifstream ifs(labelsFile);
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
	map< int, string> labelsDict;
	std::vector<std::string> list = obj["labels"].getMemberNames();

    for(size_t iter = 0; iter < list.size(); iter++ ){ 
        labelsDict.insert({ stoi(list[iter]), obj["labels"][list[iter]].asString()});
        cout << list[iter] << " " << labelsDict.at(stoi(list[iter]))<< endl;
    }
	labelsDict.insert({ -1, "Nieznany"});
	Mat frame;

	cv::cuda::GpuMat frameGpu, grayFrame, faceResized, facesBuf, descriptorGpu;
	vector<float> descriptors;
	vector<Rect> faces;
	vector<int> detectionVector(30, -1);

	Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create();

	Py_Initialize();
	PyObject *pName, *pModule, *pFunc, *sysPath, *pArgs, *pValue, *pDict;
	sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_FromString("../voice"));
    pName = PyUnicode_FromString("RunTest");
    pModule = PyImport_Import(pName);

    if (pModule == NULL) {
        std::cout << "Błąd: Nie znaleziono skryptu Python\n";
	}
	pDict = PyModule_GetDict(pModule);
	pFunc = PyDict_GetItem(pDict, PyUnicode_FromString("main"));
	
	bool mustTestVoice = false;
	bool identifyIsDone = false;
	int index = 0;
	int predictionIndex = 0;
	int predictionFreqValue =-1;

	while (true)
	{
		TickMeter timer;
		capture >> frame;
		
		if (identifyIsDone){
			usleep(5000000);
			identifyIsDone = false;
		}
		else if (!frame.empty()) {

    		timer.start();
			frameGpu.upload(frame);

			cv::cuda::cvtColor(frameGpu, grayFrame, COLOR_BGR2GRAY);
	
			faceCascade->detectMultiScale(grayFrame, facesBuf);
			faceCascade->convert(facesBuf, faces);

			for (int i = 0; i < faces.size(); i++)
			{	
				cv::cuda::GpuMat face = grayFrame(faces[i]);
				cv::cuda::resize(face, faceResized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_LINEAR);
	
				hog->compute(faceResized, descriptorGpu);
				descriptorGpu.download(descriptors);

				Mat matrixHOG(1, descriptors.size(), CV_32FC1);

				for (int i = 0; i < descriptors.size(); i++){
					matrixHOG.at<float>(0, i) = descriptors.at(i);
				}
				
				int imagePrediction = svm->predict(matrixHOG);
				rectangle(frame, faces[i], fontColor, 2);

				detectionVector[predictionIndex] = imagePrediction;

				predictionIndex++;
				if( predictionIndex == 30){
					predictionFreqValue = calculateFrequency(detectionVector);
					predictionIndex = 0;
				}
				else {
					predictionFreqValue = -1;
				}
				
				if(predictionFreqValue != -1 || mustTestVoice)
				{	
					if(!mustTestVoice){
						putText(frame, "Trwa identyfikacja mowcy...", Point(2, frame.rows - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2, 16);
						mustTestVoice = true; 
					}
					else {
						pValue = NULL;
						bool checkVoice = true;
						string voicePredictionLabel;

						while(pValue == NULL){

							if(pFunc != NULL && mustTestVoice == true){
								clog << "Trwa identyfikacja mówcy...";
								mustTestVoice = false;
								pValue = PyObject_CallObject(pFunc, NULL);
								voicePredictionLabel = _PyUnicode_AsString(pValue);
								clog << "...zakończono" << endl;
								std::cout << voicePredictionLabel<< std::endl;
							} 
							else {
								std::cout << "Nie znaleziono funkcji w pliku .py\n";
							}
						}

						string imagePredictionLabel = labelsDict.at(imagePrediction);
						clog << "Wyniki identyfikacji:" << endl;
						clog << "Mówcy: " << voicePredictionLabel<< "\tTwarzy: " << imagePredictionLabel << "\tWartość z pliku JSON: " << imagePrediction << endl;

						if(voicePredictionLabel== imagePredictionLabel){
							
							std::vector<std::string> vectorLabel;
							split( imagePredictionLabel, vectorLabel);

							for(size_t s = 0; s <vectorLabel.size(); s++ ){
								rectangle(frame, Rect(faces[i].x -1, faces[i].y + faces[i].height + 1 + (faces[i].height * 0.2 * s), faces[i].width +2, faces[i].height * 0.2), fontColor, -1);
								putText(frame, vectorLabel[s], Point(faces[i].x + 1, faces[i].y + faces[i].height * 1.13 + (faces[i].height * 0.2 * s)), FONT_HERSHEY_SIMPLEX, (float)faces[i].height/190, Scalar(255, 255, 255), 0.5, 1);				
							}
							putText(frame, "Identyfikacja zakonczona sukcesem", Point(2, frame.rows - 10), FONT_HERSHEY_SIMPLEX, 0.8, fontColor, 2, 16);
						}
						else {
							putText(frame, "Wyniki identyfikacji nie sa zgodne", Point(2, frame.rows - 55), FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2, 16);
							putText(frame, "Rozpoznanie twarzy: " + imagePredictionLabel, Point(2, frame.rows - 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1, 16);
							putText(frame, "Rozpoznanie glosu: " + voicePredictionLabel, Point(2, frame.rows - 7), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 1, 16);
						}
							
						identifyIsDone = true;
					}  
				}	
			}

			if(index == 30) index=0;

			string name = "/tmp/ramdisk/"+to_string(index)+".jpg"; 
			imwrite(name, frame);
			index++;
	
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
	Py_Finalize();
}
	