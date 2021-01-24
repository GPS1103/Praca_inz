#include <iostream>
#include "recognitionGpu1.hpp"

int main(int argc, char * argv[])
{
	
	if (argc == 1){
		int choice;
		cout << "1. Dodawanie próbek i trening\n";
		cout << "2. Trening\n";
		cout << "Wybierz jedną: ";
		cin >> choice;
		switch (choice)
		{
				
		case 1:
			makeAndSaveImages();
			trainSVM();
			break;
		case 2:
			trainSVM();
			break;
		default:
			return 0;
		}
	}
	else {	
		string s = argv[1];
		faceDetect(s);
	}	
 
	return 0;
}
