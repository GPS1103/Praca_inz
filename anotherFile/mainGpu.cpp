#include <iostream>
#include "recognitionGpu.hpp"

int main()
{
	int choice;
	cout << "1. Rozpoznawanie twarzy\n";
	cout << "2. Dodawanie i trening\n";
	cout << "Wybierz jedną: ";
	cin >> choice;
	switch (choice)
	{
	case 1:
		faceDetect();
		break;
	case 2:
		makeAndSaveImages();
		eigenClassifierTraining();
		break;
	default:
		return 0;
	}
 
	return 0;
}