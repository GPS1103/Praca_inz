
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#include <glob.h>
#include <sys/types.h>
#include <dirent.h>
#include <cmath>
#include <jsoncpp/json/json.h>
// #include <jsoncpp/json/writer.h>
// #include <jsoncpp/json/value.h>
// #include <jsoncpp/json/features.h>

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

void convert() {
    vector<Rect> faces; 
    CascadeClassifier faceCascade;
    faceCascade.load("/home/jetson/Testy/face_c++/haarcascade_frontalface_default.xml");
	//vector<cv::String> directories;//= listOfDirectories();
	String directory = "/home/jetson/Testy/negativeTest/";
    vector<cv::String> photoDir;
    String folderPath = "/home/jetson/Testy/negative/";
    
    //int imgCounter = 0;
	//size_t countDirectories = directories.size();
	//for(size_t dir = 0; dir < countDirectories; dir++){

		glob(directory, photoDir, true);
		size_t countPhotos = photoDir.size();
        clog << "Liczba zdjęć: " <<  countPhotos << endl;
        Mat gray, crop, detectedFace;
        clog << "Rozpoczęcie detekcji i konwersji...";
		for (size_t photo = 0; photo < countPhotos; photo++)
		{	
            Mat image = imread(photoDir[photo], IMREAD_COLOR );
           // CV_Assert( positive_count < labels.size() );
            if(image.empty())
            {
                std::cout << "Could not read the image: " << image << std::endl;
             //   return 1;
            }
            
			cvtColor(image, gray, COLOR_BGR2GRAY);

            faceCascade.detectMultiScale(gray, faces, 1.05, 5, 0, Size(200, 200), Size(280,280));

            for (size_t iter = 0; iter < faces.size(); iter++)
            { 
                detectedFace = gray(faces[0]);
                resize(detectedFace, crop, Size(64, 128), 0, 0, INTER_LINEAR);
                stringstream nameStream;
                nameStream << folderPath.c_str() << photo << ".jpg";
                string imageName = nameStream.str();
                imwrite(imageName, crop);
            }
    	}
        clog << "...zakończono" << endl;
	//}	
}

void json(){
    
    // ifstream ifs("labels.json");
    // Json::Reader reader;
    // Json::Value obj;
    // reader.parse(ifs, obj);
    // cout << obj["lastname"] << endl;
    Json::Value event;   
    //Json::Write write;
    // Json::Value vec(Json::arrayValue);
    // vec.append(Json::Value("Ela"));
    // vec.append(Json::Value("Adi"));
    // vec.append(Json::Value("Mina"));

    event["labels"][std::to_string(0)] = "Ela";
    Json::StyledWriter styledWriter;
   // cout << styledWriter.write(event);
    ofstream myfile;
    myfile.open ("labels.json");
    myfile << styledWriter.write(event);
    myfile.close();
    // event["competitors"]["away"]["code"] = 89223;
    // event["competitors"]["away"]["name"] = "Aston Villa";
    //event["labels"]=vec;

   //std::cout << event << std::endl;
}

void read(){
    ifstream ifs("labels.json");
    Json::Reader reader;
    Json::Value obj;
    reader.parse(ifs, obj);
    //cout << obj["labels"]["0"] << obj["labels"]. << endl;
    
	map< int, string> labelsDict;// = {
    //std::map<string, string>::iterator it = labelsDict.begin();
    //size_t lab = obj["labels"].getMemberNames().size();
    std::vector<std::string> list = obj["labels"].getMemberNames();
    
    for(size_t iter = 0; iter < list.size(); iter++ ){
        cout << list[iter] << endl; 
        labelsDict.insert({ stoi(list[iter]), obj["labels"][list[iter]].asString()});
        cout << labelsDict.at(stoi(list[iter]))<< endl;
    }
    // cout << obj["labels"].getMemberNames().size() << endl;
    //for (auto &const id : obj["labels"].getMemberNames()) {
        //int l = id - '0';
        //std::cout << id << std::endl;
       //
    //}
	// 	{ 0, "xss" },
	// 	{ 1, "2sdsd" },
	// 	{ 2, "3sdsdsd" }
	// };
	//
}
int main(){

    //convert();
    //json();
    read();
    return 0;
}