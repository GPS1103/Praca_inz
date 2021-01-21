#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include <time.h>
#include <vector>
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
#include <jsoncpp/json/json.h>

using namespace cv;
using namespace cv::ml;
using namespace std;
bool test = true;
string path = "/home/jetson/Testy/face_c++/photos/";
vector< float > get_svm_detector( const Ptr< SVM >& svm );
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradientsList, bool use_flip );
void test_trained_detector( String obj_det_filename, String test_dir, String videofilename );

// vector< float > get_svm_detector( const Ptr< SVM >& svm )
// {
//     // get the support vectors
//     Mat sv = svm->getSupportVectors();
//     const int sv_total = sv.rows;
//     // get the decision function
//     Mat alpha, svidx;
//     double rho = svm->getDecisionFunction( 0, alpha, svidx );

//     CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
//     CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
//                (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
//     CV_Assert( sv.type() == CV_32F );

//     vector< float > hog_detector( sv.cols + 1 );
//     memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
//     hog_detector[sv.cols] = (float)-rho;
//     return hog_detector;
// }

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{

    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); 
    trainData = Mat( rows, cols, CV_32FC1 );
    clog << "train " << train_samples.size()<< endl;
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
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

void getImages(vector<Mat>& images, vector<int>& labels) {

	vector<cv::String> directories = listOfDirectories();
	vector<cv::String> photoDir;
    Json::Value jsonEvent; 
    Json::StyledWriter jsonWriter;

    jsonEvent["labels"][std::to_string(-1)] = "Nieznany";

	size_t countDirectories = directories.size();
	for(size_t dir = 0; dir < countDirectories; dir++)
    {
        glob(path + directories[dir] + "/", photoDir, true);
        size_t countPhotos = photoDir.size();
        jsonEvent["labels"][std::to_string(dir)] = directories[dir];

        Mat imgNoising, imgDenoising;
        float mean = (10,13,18);
        float sigma = (1,5,15);
            
        clog << "Nazwa folderu: " << directories[dir] << " Etykieta: " << dir << endl;
        for (size_t photo = 0; photo < countPhotos; photo++)
        {	
            Mat img = imread(photoDir[photo]);
            Mat noise = Mat(img.size(), img.type());
            imgNoising = img.clone();

            images.push_back(img);
            labels.push_back(dir);
            //if(test) imwrite("/home/jetson/Testy/test/img.jpg", img);
            
            // GaussianBlur(img, imgDenoising, Size(9, 9), 0);
            // images.push_back(imgDenoising);
            // labels.push_back(dir);
            // if(test) imwrite("/home/jetson/Testy/test/denoise.jpg", imgDenoising);

            // cv::randn(noise, mean, sigma);
            // imgNoising+= noise;
            // images.push_back(imgNoising);
            // labels.push_back(dir);
            // if(test) imwrite("/home/jetson/Testy/test/noise.jpg", imgNoising);
    
            // flip(img, img, 1);
            // images.push_back(img);
            // labels.push_back(dir);
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
	}

    ofstream jsonFile;
    jsonFile.open("labels.json");	
    jsonFile << jsonWriter.write(jsonEvent);
    jsonFile.close();
}

void getNegativeImages(vector<Mat>& images, vector<int>& labels) {

    String neg_dir = "/home/jetson/Testy/negative";
    vector<cv::String> photoDir;

	glob(neg_dir, photoDir, true);
	size_t countPhotos = photoDir.size();

    Mat imgNoising, imgDenoising;
    float mean = (10,15,20);
    float sigma = (1,5,25);

	for (size_t photo = 0; photo < countPhotos; photo++)
	{	
		Mat img = imread(photoDir[photo]);
        Mat noise = Mat(img.size(), img.type());
        imgNoising = img.clone();

		images.push_back(img);
		labels.push_back(-1);
        //if(test) imwrite("/home/jetson/Testy/test/img.jpg", img);
           
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
            
        // test = false;
		}

}

// void load_images( const String & dirname, vector<Mat>& img_lst, bool showImages = false )
// {
//     vector< String > files;
//     glob( dirname, files );

//     for ( size_t i = 0; i < files.size(); ++i )
//     {
//         Mat img = imread( files[i] ); // load the image
//         if ( img.empty() )
//         {
//             cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
//             continue;
//         }

//         if ( showImages )
//         {
//             imshow( "image", img );
//             waitKey( 1 );
//         }
//         img_lst.push_back( img );
//     }
// }

// void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
// {
//     Rect box;
//     box.width = size.width;
//     box.height = size.height;

//     srand( (unsigned int)time( NULL ) );

//     for ( size_t i = 0; i < full_neg_lst.size(); i++ )
//         if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
//         {
//             box.x = rand() % ( full_neg_lst[i].cols - box.width );
//             box.y = rand() % ( full_neg_lst[i].rows - box.height );
//             Mat roi = full_neg_lst[i]( box );
//             neg_lst.push_back( roi.clone() );
//         }
// }

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradientsList )
{
    HOGDescriptor hog;//(Size(64,128), Size(16,16), Size(8,8), Size(2,2), 9 )
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;

    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            // Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
            //               ( img_lst[i].rows - wsize.height ) / 2,
            //               wsize.width,
            //               wsize.height);
        
            hog.compute( img_lst[i], descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradientsList.push_back( Mat( descriptors ).clone() );
            
        }
    }
}

// void test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
// {
//     cout << "Testing trained detector..." << endl;
//     HOGDescriptor hog;
//     hog.load( obj_det_filename );

//     vector< String > files;
//     glob( test_dir, files );

//     int delay = 0;
//     VideoCapture cap;

//     if ( videofilename != "" )
//     {
//         if ( videofilename.size() == 1 && isdigit( videofilename[0] ) )
//             cap.open( videofilename[0] - '0' );
//         else
//             cap.open( videofilename );
//     }

//     obj_det_filename = "testing " + obj_det_filename;
//     namedWindow( obj_det_filename, WINDOW_NORMAL );

//     for( size_t i=0;; i++ )
//     {
//         Mat img;

//         if ( cap.isOpened() )
//         {
//             cap >> img;
//             delay = 1;
//         }
//         else if( i < files.size() )
//         {
//             img = imread( files[i] );
//         }

//         if ( img.empty() )
//         {
//             return;
//         }

//         vector< Rect > detections;
//         vector< double > foundWeights;

//         hog.detectMultiScale( img, detections, foundWeights );
//         for ( size_t j = 0; j < detections.size(); j++ )
//         {
//             Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
//             rectangle( img, detections[j], color, img.cols / 400 + 1 );
//         }

//         imshow( obj_det_filename, img );

//         if( waitKey( delay ) == 27 )
//         {
//             return;
//         }
//     }
// }

void trainSVM(){

    String neg_dir = "/home/jetson/Testy/negative";
    String obj_det_filename = "/home/jetson/Testy/svm.xml";

    int detector_width = 64;
    int detector_height = 128;

    vector< Mat > imagesVector, full_neg_lst, neg_lst, gradientsList;
    vector< int > labels;

    clog << "Inicjalizacja pozytywnych zdjęć..." << endl ;
    getImages(imagesVector, labels);

    int posImages = imagesVector.size();
    CV_Assert(posImages != 0);
    // if ( imagesVector.size() > 0 )
    // {
    clog << "...załadowano " << posImages << " zdjęć." << endl;
    // }
    // else
    // {
    //     clog << "nie znaleziono zdjęć" <<endl;
        
    // }

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

    Mat train_data;
    convert_to_ml( gradientsList, train_data );

    clog << "Trenowanie SVM..." << endl << "wiersze: " << train_data.rows << endl << "kolumny: " << train_data.cols << endl;
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
   // svm->setCoef0( 0.0 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-3 ) );
    //svm->setGamma( 10 );
    svm->setKernel( SVM::LINEAR );
   // svm->setNu( 0.5 );
    //svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 20 ); // From paper, soft classifier
    svm->setType(cv::ml::SVM::C_SVC);
    //svm->setKernel(cv::ml::SVM::POLY);
    svm->setGamma(100);    
    svm->setDegree(0.65);

    convert_to_ml( gradientsList, train_data );
    
    svm->train( train_data, ROW_SAMPLE, labels );

    clog << "...zakończono " << endl;
    

    if (true)
    {
        gradientsList.clear();
		clog << "Ponowne obliczanie HOG dla wszystkich zdjęć...";
        computeHOGs( imagesVectorSize, imagesVector, gradientsList);
        int positive_count = gradientsList.size();
        clog << "...zakończono ( liczba zdjęć : " << imagesVector.size() << " )" << endl;

        clog << "Ponowne trenowanie SVM...";
        convert_to_ml( gradientsList, train_data );
        svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6 ) );
        svm->train( train_data, ROW_SAMPLE, labels );
        clog << "...zakończono" << endl;
    }
    svm->save(obj_det_filename);

}

int main( int argc, char** argv )
{
    trainSVM();
    return 0;
}