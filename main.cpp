#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

cv::Mat src, src_gray, combined_dst;
cv::VideoCapture cap;

int erosion_active = 0;
int erosion_elem = 0;
int erosion_size = 0;
int dilation_active = 0;
int dilation_elem = 0;
int dilation_size = 0;
int resize_active = 0;
int resize_factor = 100;  // Start with 100% to avoid initial resizing
int brightness_active = 0;
int brightness_factor = 100;  // 100% means no change
int canny_active = 0;
int lowThreshold = 50;  // Initial low threshold for Canny
int highThreshold = 150;  // Initial high threshold for Canny

int const max_elem = 2;
int const max_kernel_size = 21;
int const max_resize_factor = 200;
int const max_brightness_factor = 200;
int const max_lowThreshold = 100;
int const max_highThreshold = 300;

void UpdateImage(int, void*);
void StitchImages(vector<String>& imagePaths);
bool LoadImage(const string& path);
bool LoadVideo(const string& path);
void ProcessFrame(Mat& frame);
void CreateTrackbars();

int main(int argc, char** argv) {
    string filePath;
    char fileType;

    cout << "Enter 'i' for image or 'v' for video: ";
    cin >> fileType;
    cin.ignore(); // To ignore the newline character after entering the file type

    cout << "Enter the path of the file you want to process: ";
    getline(cin, filePath);

    bool isVideo = (fileType == 'v' || fileType == 'V');

    if (isVideo) {
        if (!LoadVideo(filePath)) {
            return -1;
        }
    }
    else {
        if (!LoadImage(filePath)) {
            return -1;
        }
    }

    namedWindow("Combined Demo", WINDOW_AUTOSIZE);

    // Increase the size of the window to fit long trackbar titles
    int control_window_width = 600; // Adjust this value as needed to fit your titles
    int control_window_height = 600; // Adjust this value as needed for the number of trackbars
    resizeWindow("Combined Demo", control_window_width, control_window_height);

    CreateTrackbars();
    UpdateImage(0, 0);

    if (isVideo) {
        while (true) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            ProcessFrame(frame);

            imshow("Combined Demo", combined_dst);

            if (waitKey(30) == 27) { // ESC key
                break;
            }
        }
    }
    else {
        vector<String> imagePaths;
        string moreImages;
        while (true) {
            cout << "Enter the path of an image to stitch (or 'done' to finish): ";
            getline(cin, moreImages);
            if (moreImages == "done") break;
            imagePaths.push_back(moreImages);
        }

        StitchImages(imagePaths);

        while (true) {
            int key = waitKey(0);
            if (key == 's' || key == 'S') {
                imwrite("modified_image.png", combined_dst);
                cout << "Image saved as modified_image.png" << endl;
            }
            else if (key == 27) { // ESC key
                break;
            }
        }
    }

    return 0;
}

bool LoadImage(const string& path) {
    src = imread(path, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        return false;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    return true;
}

bool LoadVideo(const string& path) {
    cout << "Trying to open video: " << path << endl; // Debug statement

    cap.open(path);
    if (!cap.isOpened()) {
        cout << "Could not open or find the video!\n" << endl;
        return false;
    }

    cout << "Video opened successfully!" << endl; // Debug statement
    return true;
}

void CreateTrackbars() {
    createTrackbar("Erosion Active: 0: Off 1: On", "Combined Demo", &erosion_active, 1, UpdateImage);
    createTrackbar("Erosion Element: 0: Rect 1: Cross 2: Ellipse", "Combined Demo", &erosion_elem, max_elem, UpdateImage);
    createTrackbar("Erosion Kernel size: 2n +1", "Combined Demo", &erosion_size, max_kernel_size, UpdateImage);

    createTrackbar("Dilation Active: 0: Off 1: On", "Combined Demo", &dilation_active, 1, UpdateImage);
    createTrackbar("Dilation Element: 0: Rect 1: Cross 2: Ellipse", "Combined Demo", &dilation_elem, max_elem, UpdateImage);
    createTrackbar("Dilation Kernel size: 2n +1", "Combined Demo", &dilation_size, max_kernel_size, UpdateImage);

    createTrackbar("Resize Active: 0: Off 1: On", "Combined Demo", &resize_active, 1, UpdateImage);
    createTrackbar("Resize factor: %", "Combined Demo", &resize_factor, max_resize_factor, UpdateImage);

    createTrackbar("Brightness Active: 0: Off 1: On", "Combined Demo", &brightness_active, 1, UpdateImage);
    createTrackbar("Brightness factor: %", "Combined Demo", &brightness_factor, max_brightness_factor, UpdateImage);

    createTrackbar("Canny Active: 0: Off 1: On", "Combined Demo", &canny_active, 1, UpdateImage);
    createTrackbar("Low Threshold:", "Combined Demo", &lowThreshold, max_lowThreshold, UpdateImage);
    createTrackbar("High Threshold:", "Combined Demo", &highThreshold, max_highThreshold, UpdateImage);
}

void UpdateImage(int, void*) {
    if (!src.empty()) {
        ProcessFrame(src);
        imshow("Combined Demo", combined_dst);
    }
}

void ProcessFrame(Mat& frame) {
    Mat current_dst = frame.clone();
    Mat temp_dst;

    // Erosion
    if (erosion_active) {
        int erosion_type = erosion_elem == 0 ? MORPH_RECT : (erosion_elem == 1 ? MORPH_CROSS : MORPH_ELLIPSE);
        Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
        erode(current_dst, temp_dst, element);
        current_dst = temp_dst.clone();
    }

    // Dilation
    if (dilation_active) {
        int dilation_type = dilation_elem == 0 ? MORPH_RECT : (dilation_elem == 1 ? MORPH_CROSS : MORPH_ELLIPSE);
        Mat element = getStructuringElement(dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));
        dilate(current_dst, temp_dst, element);
        current_dst = temp_dst.clone();
    }

    // Resize
    if (resize_active) {
        double scale = resize_factor / 100.0;
        resize(current_dst, temp_dst, Size(), scale, scale, INTER_LINEAR);
        current_dst = temp_dst.clone();
    }

    // Adjust Brightness
    if (brightness_active) {
        double alpha = brightness_factor / 100.0;
        double beta = (brightness_factor - 100) * 2.55;
        current_dst.convertTo(temp_dst, -1, alpha, beta);
        current_dst = temp_dst.clone();
    }

    // Canny Edge Detection
    if (canny_active) {
        cvtColor(current_dst, temp_dst, COLOR_BGR2GRAY);
        Canny(temp_dst, current_dst, lowThreshold, highThreshold, 3);
    }

    combined_dst = current_dst;
}

void StitchImages(vector<String>& imagePaths) {
    vector<Mat> images;
    for (const auto& imagePath : imagePaths) {
        Mat img = imread(imagePath);
        if (img.empty()) {
            cout << "Could not open or find the image at path: " << imagePath << endl;
            return;
        }
        images.push_back(img);
    }
    if (images.size() < 2) {
        cout << "Need at least two images to perform stitching." << endl;
        return;
    }

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Stitcher::Status status = stitcher->stitch(images, pano);

    if (status != Stitcher::OK) {
        cout << "Can't stitch images, error code = " << status << endl;
        return;
    }

    namedWindow("Stitched Image", WINDOW_AUTOSIZE);
    imshow("Stitched Image", pano);
    combined_dst = pano;

    while (true) {
        int key = waitKey(0);
        if (key == 's' || key == 'S') {
            imwrite("stitched_image.png", combined_dst);
            cout << "Stitched image saved as stitched_image.png" << endl;
        }
        else if (key == 27) { // ESC key
            break;
        }
    }
}

//C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\mainWhali.png
//C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\image1.jpg
 //C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\image2.jpg
 // C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\image3.jpg
// //C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\mainWhali.png
//C:\Users\victo\Documents\MultimediaApplicationProjects\OpenCvWorkshop\OpenCVTP2\x64\Release\video.mp4
//C:\\Users\\victo\\Documents\\MultimediaApplicationProjects\\OpenCvWorkshop\\OpenCVTP2\\x64\\Release\\video.mp4

