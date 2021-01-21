  stringstream ss;
string name = "Vehicle_";
string type = ".jpg";

int num_train_images = 29;      //29 images will be used to train the SVM
int image_area = 150 * 200;
Mat training_mat(num_train_images, image_area, CV_32FC1);   // Creates a 29 rows by 30000 columns... 29 150x200 images will be put into 1 row per image

//Converts 29 2D images into a really long row per image
for (int file_count = 1; file_count < (num_train_images + 1); file_count++) 
{
    ss << name << file_count << type;       //'Vehicle_1.jpg' ... 'Vehicle_2.jpg' ... etc ...
    string filename = ss.str();
    ss.str("");

    Mat training_img = imread(filename, 1);     //Reads the training images from the folder
    int ii = 0;                                 //Scans each column
    for (int i = 0; i < training_img.rows; i++) 
    {
        for (int j = 0; j < training_img.cols; j++)
        {
            training_mat.at<float>(file_count - 1, ii) = training_img.at<uchar>(i, j);  //Fills the training_mat with the read image
            ii++; 
        }
    }
}

//Labels are used as the supervised learning portion of the SVM. If it is a 1, its an SUV test image. -1 means a sedan. 
float labels[29] = { 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1 };

//Place the labels into into a 29 row by 1 column matrix. 
Mat labels_mat(num_train_images, 1, CV_32FC1, labels);

cout << "Beginning Training..." << endl;
  //Set SVM Parameters (not sure about these values, but just wanna see something)
Ptr<SVM> svm = SVM::create();
svm->setType(SVM::C_SVC);
svm->setKernel(SVM::POLY);
svm->setC(50);
svm->setGamma(100);
svm->setDegree(.65);
//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

cout << "Parameters Set..." << endl;

svm->train(HOGFeat_train, ROW_SAMPLE, labels_mat);

Mat SV = svm->getSupportVectors();
Mat USV = svm->getUncompressedSupportVectors();

cout << "Support Vectors: " << SV.rows << endl;
cout << "Uncompressed Support Vectors: " << USV.rows << endl;

cout << "Training Successful" << endl;

waitKey(0);

//TESTING PORTION

cout << "Begin Testing..." << endl;

int num_test_images = 10;
Mat HOGFeat_test(1, derSize, CV_32FC1); //Creates a 1 x descriptorSize Mat to house the HoG features from the test image

for (int file_count = 1; file_count < (num_test_images + 1); file_count++)
{

    test << nameTest << file_count << type;     //'Test_1.jpg' ... 'Test_2.jpg' ... etc ...
    string filenameTest = test.str();
    test.str("");

    Mat test_image = imread(filenameTest, 0);           //Read the file folder

    HOGDescriptor hog_test;// (Size(64, 64), Size(32, 32), Size(16, 16), Size(32, 32), 9, 1, -1, 0, .2, 1, 64, false);
    vector<float> descriptors_test;
    vector<Point> locations_test;

    hog_test.compute(test_image, descriptors_test, Size(64, 64), Size(0, 0), locations_test);

    for (int i = 0; i < descriptors_test.size(); i++)
        HOGFeat_test.at<float>(0, i) = descriptors_test.at(i);

    namedWindow("Test Image", CV_WINDOW_NORMAL);
    imshow("Test Image", test_image);

    //Should return a 1 if its an SUV, or a -1 if its a sedan
    float result = svm->predict(HOGFeat_test);

    if (result <= 0)
        cout << "Sedan" << endl;
    else
        cout << "SUV" << endl;

    cout << "Result: " << result << endl;

    waitKey(0);
}