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
#include <map>
#include <string>
#include <jsoncpp/json/json.h>
#include <curl/curl.h>

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

int imgHeight = 128; 
int imgWidth = 64;
string cascade = "Classifier/haarcascade_frontalface_default.xml";
cv::Ptr<cv::cuda::CascadeClassifier> faceCascade = cv::cuda::CascadeClassifier::create(cascade);
string positiveDirectory = "photos/positive/";
string negDirectory = "photos/negative";
string svmClassifier = "Classifier/svm.xml";
string labelsFile = "labels.json";
string name;

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
    svm->setDegree( 2 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER, 1e3, 1e-6) );
    svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	//svm->setC(20);
	//svm->setGamma(CvParamGrid_gamma);
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
	svm->trainAuto( trainData, ROW_SAMPLE, labels,  2, CvParamGrid_C, CvParamGrid_gamma );
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

void checkClassifier(){
    clog << "Testowanie klasyfikatora" << endl;
	Ptr< cv::ml::SVM > svm = Algorithm::load<cv::ml::SVM>(svmClassifier);

	Mat SV = svm->getSupportVectors();
	Mat USV = svm->getUncompressedSupportVectors();

	cout << "Support Vectors: " << SV.rows << endl;
	cout << "Uncompressed Support Vectors: " << USV.rows << endl;

	ifstream ifs(labelsFile);
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
	map< int, string> labelsDict;
	std::vector<std::string> list = obj["labels"].getMemberNames();

    for(size_t iter = 0; iter < list.size(); iter++ ){
        cout << list[iter] << endl; 
        labelsDict.insert({ stoi(list[iter]), obj["labels"][list[iter]].asString()});
        cout << labelsDict.at(stoi(list[iter]))<< endl;
    }

    vector< Mat > imagesVector, gradientsList;
    vector< int > labels;

    positiveDirectory = "photos_offline/positive/";
    negDirectory = "photos_offline/negative";

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

	Mat frame;

	cv::cuda::GpuMat frameGpu, grayFrame, faceResized, facesBuf, descriptorGpu;
	vector<float> descriptors;
	vector<Rect> faces;

	Ptr<cv::cuda::HOG> hog = cv::cuda::HOG::create();
    if (!imagesVector.empty()) {

	    for ( int iter = 0; iter < imagesVector.size(); iter++)
	    {

			//cv::cuda::GpuMat face = grayFrame(faces[i]);
			//cv::cuda::resize(face, faceResized, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_LINEAR);
            frameGpu.upload(imagesVector[iter]);
            cv::cuda::resize(frameGpu, frameGpu, Size(imgWidth, imgHeight), 1.0, 1.0, INTER_LINEAR);
            
			hog->compute(frameGpu, descriptorGpu);
            clog << "hog computed" << endl;
			descriptorGpu.download(descriptors);
			Mat matrixHOG(1, descriptors.size(), CV_32FC1);
			for (int i = 0; i < descriptors.size(); i++){
				matrixHOG.at<float>(0, i) = descriptors.at(i);
			}
			
			int prediction = svm->predict(matrixHOG);
			//rectangle(frame, faces[i], fontColor, 2);
            clog << "Json: " << labels[iter] << " " << "Predict: " << prediction << endl;
			
			// if(prediction != -1)
			// {	
				
			// 	string labelInfo = labelsDict.at(prediction);
			// 	std::vector<std::string> vectorLabel;
    		// 	split( labelInfo, vectorLabel);
			// 	clog << vectorLabel[0] << " predykcja: " << prediction << endl;
			// 	for(size_t s = 0; s <vectorLabel.size(); s++ ){
			// 		rectangle(frame, Rect(faces[i].x -1, faces[i].y + faces[i].height + 1 + (faces[i].height * 0.2 * s), faces[i].width +2, faces[i].height * 0.2), fontColor, -1);
			// 		putText(frame, vectorLabel[s], Point(faces[i].x + 1, faces[i].y + faces[i].height * 1.13 + (faces[i].height * 0.2 * s)), FONT_HERSHEY_SIMPLEX, (float)faces[i].height/190, Scalar(255, 255, 255), 0.5, 1);				
	 	}
			
	}
}


int main(){

    trainSVM();
    checkClassifier();
}
