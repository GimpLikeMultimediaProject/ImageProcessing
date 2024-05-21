#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

cv::Mat src, erosion_dst, dilation_dst, resize_dst;

int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int resize_factor = 10;

int const max_elem = 2;
int const max_kernel_size = 21;
int const max_resize_factor = 200;

void Erosion(int, void*);
void Dilation(int, void*);
void ResizeImage(int, void*);

int main(int argc, char** argv) {

    CommandLineParser parser(argc, argv, "{@input | C:/Users/victo/Documents/MultimediaApplicationProjects/OpenCvWorkshop/OpenCVTP2/x64/Release/mainWhali.png | input image}");
    src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
    namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
    namedWindow("Resize Demo", WINDOW_AUTOSIZE);

    moveWindow("Dilation Demo", src.cols, 0);
    moveWindow("Resize Demo", 2 * src.cols, 0);

    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
        &erosion_elem, max_elem,
        Erosion);
    createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
        &erosion_size, max_kernel_size,
        Erosion);
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
        &dilation_elem, max_elem,
        Dilation);
    createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
        &dilation_size, max_kernel_size,
        Dilation);
    createTrackbar("Resize factor:\n %", "Resize Demo",
        &resize_factor, max_resize_factor, ResizeImage);

    Erosion(0, 0);
    Dilation(0, 0);
    ResizeImage(0, 0);

    waitKey(0);
    return 0;
}
//cv::Mat img = cv::imread("C:/Users/victo/Documents/MultimediaApplicationProjects/OpenCvWorkshop/OpenCVTP2/x64/Release/mainWhali.png");


//cv::imshow("windows", img);
//cv::waitKey(0); //always imshow and waitkey together

void Erosion(int, void*)
{
    int erosion_type = 0;
    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(erosion_type,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size));
    erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
}


void Dilation(int, void*)
{
    int dilation_type = 0;
    if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
    else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
    else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(dilation_type,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    dilate(src, dilation_dst, element);
    imshow("Dilation Demo", dilation_dst);
}

void ResizeImage(int, void*)
{
    double scale = resize_factor / 100.0; //convert percentage to a scale factor
    resize(src, resize_dst, Size(), scale, scale, INTER_LINEAR);
    imshow("Resize Demo", resize_dst);
}