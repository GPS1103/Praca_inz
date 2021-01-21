#include <iostream>
#include "recognitionGpu1.hpp"

int main()
{
	int choice;
	cout << "1. Rozpoznawanie twarzy\n";
	cout << "2. Dodawanie próbek i trening\n";
	cout << "3. Trening\n";
	cout << "Wybierz jedną: ";
	cin >> choice;
	switch (choice)
	{
	case 1:
		faceDetect();
		break;
	case 2:
		makeAndSaveImages();
		trainSVM();
		break;
	case 3:
		trainSVM();
		break;
	default:
		return 0;
	}
 
	return 0;
}