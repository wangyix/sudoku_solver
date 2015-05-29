//#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/*extern "C" {
    JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tuturial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba);

    JNIEXPORT void JNICALL Java_org_opencv_samples_tutorial2_Tutorial2Activity_FindFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)*/

Point2i twoRectsMatch(const Mat& integral,
    const Point2i& r1_min, const Point2i& r1_max,
    const Point2i& r2_min, const Point2i& r2_max,
    const Point2i& s1_min, const Point2i& s1_max,
    const Point2i& s2_min, const Point2i& s2_max,
    const Point2i& center_min, const Point2i& center_max);

int main(void)
{
    Mat im;
    string filename = "../sudoku2.jpg";
    im = imread(filename, CV_LOAD_IMAGE_COLOR);

    if (!im.data) {
        cout << "Could not open " << filename << endl;
        return -1;
    }
    namedWindow("original image", WINDOW_AUTOSIZE);
    imshow("original image", im);

    const int IM_ROWS = im.rows;
    const int IM_COLS = im.cols;
    const double IM_SQRT_RES = sqrt(IM_ROWS * IM_COLS);

    int64 A, B, C, D;
    const double TICK_FREQ_MS = getTickFrequency()*0.001f;

    C = getTickCount();

    // binarize image with adaptive thresholding

    A = getTickCount();
    
    Mat im_gray;
    cvtColor(im, im_gray, CV_RGB2GRAY);
    // mean of nxn neighborhood of pixel minus offset is used as threshold
    Mat im_bin;
    {
        int blockSize = (int)((1.0 / 16.0) * 0.5 * IM_SQRT_RES) * 2 + 1;
        cout << "adaptive thresholding blocksize " << blockSize << endl;
        int meanOffset = 12;
        adaptiveThreshold(im_gray, im_bin, 255, ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY_INV, blockSize, meanOffset);
    }
    B = getTickCount();
    cout << "binarize: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    namedWindow("thesholded image", WINDOW_AUTOSIZE);
    imshow("thesholded image", im_bin);



    // remove everything except the largest blob, assumed to be the sudoku grid
    A = getTickCount();
    Mat im_binblob;
    im_bin.copyTo(im_binblob);
    {
        int maxArea = 0;
        Point maxAreaPoint;
        for (int y = 0; y < IM_ROWS; y++) {
            uchar *row = im_binblob.ptr(y);
            for (int x = 0; x < IM_COLS; x++) {
                if (row[x] > 128) {
                    int area = floodFill(im_binblob, Point(x, y), Scalar(128));
                    if (area > maxArea) {
                        maxArea = area;
                        maxAreaPoint = Point(x, y);
                    }
                }
            }
        }
        floodFill(im_binblob, maxAreaPoint, Scalar(255));
        threshold(im_binblob, im_binblob, 254, 255, CV_THRESH_BINARY);
    }
    /*{
        vector<vector<Point>> contours;
        findContours(im_binblob, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
        
        // find largest blob
        int maxArea = 0;
        int maxIndex = -1;
        for (int i = 0; i < contours.size(); i++) {
            int area = contours[i].size();
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
        }

        im_binblob = Mat::zeros(im_binblob.size(), im_binblob.type());
        drawContours(im_binblob, contours, maxIndex, Scalar(255));
    }*/
    B = getTickCount();
    cout << "find max blob: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    namedWindow("max blob", WINDOW_AUTOSIZE);
    imshow("max blob", im_binblob);


    
    
    // find the contour of the grid
    A = getTickCount();
    Mat im_edge = Mat::zeros(IM_ROWS, IM_COLS, CV_8U);
    for (int y = 1; y < IM_ROWS - 1; y++) {
        uchar* row_above = im_binblob.ptr(y - 1);
        uchar* row = im_binblob.ptr(y);
        uchar* row_below = im_binblob.ptr(y + 1);
        uchar* edge_row = im_edge.ptr(y);
        for (int x = 1; x < IM_COLS - 1; x++) {
            uchar center = row[x];
            if (!center) {
                edge_row[x] = 0;
                continue;
            }
            uchar top = row_above[x];
            uchar bottom = row_below[x];
            uchar left = row[x - 1];
            uchar right = row[x + 1];
            if (top && bottom && left && right) {
                edge_row[x] = 0;
            } else {
                edge_row[x] = 255;
            }
        }
    }

    B = getTickCount();
    cout << "manual contour: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    //namedWindow("manual contour", CV_WINDOW_AUTOSIZE);
    //imshow("manual contour", im_edge);

    // find the four corners of the grid
    A = getTickCount();
    vector<Point2f> corners(4);
    {
        // corners are, in order:
        // min projection onto <1,1>  (top left)    minSum
        // max projection onto <1,-1> (top right)   maxDiff
        // max projection onto <1,1> (bottom right) maxSum
        // min projection onto <1,-1> (bottom left) minDiff
        int largeEnough = IM_COLS + IM_ROWS;
        int maxSum = -largeEnough, minSum = largeEnough;
        int maxDiff = -largeEnough, minDiff = largeEnough;
        for (int y = 0; y < IM_ROWS; y++) {
            uchar *row = im_edge.ptr(y);
            for (int x = 0; x < IM_COLS; x++) {
                if (row[x]) {
                    int sum = x + y;    // projection onto <1,1>
                    int diff = x - y;   // projection onto <1,-1>
                    if (sum < minSum) {
                        minSum = sum;
                        corners[0] = Point2f(x, y);
                    }
                    if (sum > maxSum) {
                        maxSum = sum;
                        corners[2] = Point2f(x, y);
                    }
                    if (diff < minDiff) {
                        minDiff = diff;
                        corners[3] = Point2f(x, y);
                    }
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        corners[1] = Point2f(x, y);
                    }
                }
            }
        }
    }
    B = getTickCount();
    cout << "find 4 corners: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;
    
    
    // draw corners
    Mat im_corners;
    im.copyTo(im_corners);
    for (int i = 0; i < 4; i++) {
        circle(im_corners, corners[i], 5, Scalar(255,0,0), 2, 8, 0);
    }
    namedWindow("4 corners", CV_WINDOW_AUTOSIZE);
    imshow("4 corners", im_corners);




    // find homography and straighten grid
    A = getTickCount();
    const int CELL_DIM = 51;    // make this odd
    const int GRID_DIM = 9 * CELL_DIM;
    const int GRID_PAD = 30;
    const int GRID_DIM_PAD = GRID_DIM + 2 * GRID_PAD;
    Mat im_grid;
    {
        vector<Point2f> dstPoints;
        dstPoints.push_back(Point2f(GRID_PAD, GRID_PAD));
        dstPoints.push_back(Point2f(GRID_PAD + GRID_DIM, GRID_PAD));
        dstPoints.push_back(Point2f(GRID_PAD + GRID_DIM, GRID_PAD + GRID_DIM));
        dstPoints.push_back(Point2f(GRID_PAD, GRID_PAD + GRID_DIM));
        Mat transform = findHomography(corners, dstPoints, 0);
        warpPerspective(im_bin, im_grid, transform, Size(GRID_DIM_PAD, GRID_DIM_PAD),
            INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
    }
    B = getTickCount();
    cout << "homography transform: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;
    
    namedWindow("grid", CV_WINDOW_AUTOSIZE);
    imshow("grid", im_grid); 


    // find best center of each grid cell with two-bar template matching

    vector<Point2i> cellCenters(81);
    Mat im_grid_thick;

    A = getTickCount();
    {
        // dilate image for thicker lines
        {
            int SE_radius = 1;
            int SE_size = SE_radius * 2 + 1;
            cout << "dilate SE size " << SE_size << endl;
            Mat dilate_SE = Mat::zeros(SE_size, SE_size, CV_8U);
            for (int y = 0; y < SE_size; y++)
                dilate_SE.ptr(y)[SE_radius] = 1;
            uchar* row = dilate_SE.ptr(SE_radius);
            for (int x = 0; x < SE_size; x++)
                row[x] = 1;
            dilate(im_grid, im_grid_thick, dilate_SE);
        }

        const int CELL_HALF_DIM = (CELL_DIM - 1) / 2;
        const int BAR_HALF_THICKNESS = 2;
        const int BAR_THICKNESS = 2 * BAR_HALF_THICKNESS;
        const int ACCEPT_WIDTH = 15;

        // bounds for horizontal bars
        Point2i hor_r1_min;
        hor_r1_min.x = -CELL_HALF_DIM;
        hor_r1_min.y = -CELL_HALF_DIM - BAR_HALF_THICKNESS;
        Point2i hor_r1_max;
        hor_r1_max.x = hor_r1_min.x + CELL_DIM;
        hor_r1_max.y = hor_r1_min.y + BAR_THICKNESS;
        Point2i hor_s1_min(hor_r1_min.x + ACCEPT_WIDTH, hor_r1_min.y);
        Point2i hor_s1_max(hor_r1_max.x - ACCEPT_WIDTH, hor_r1_max.y);

        Point2i hor_r2_min = hor_r1_min + Point2i(0, CELL_DIM);
        Point2i hor_r2_max = hor_r1_max + Point2i(0, CELL_DIM);
        Point2i hor_s2_min = hor_s1_min + Point2i(0, CELL_DIM);
        Point2i hor_s2_max = hor_s1_max + Point2i(0, CELL_DIM);
        // bounds for vertical bars (simply the transpose of horizontal bar bounds)
        Point2i vert_r1_min(hor_r1_min.y, hor_r1_min.x);
        Point2i vert_r1_max(hor_r1_max.y, hor_r1_max.x);
        Point2i vert_r2_min(hor_r2_min.y, hor_r2_min.x);
        Point2i vert_r2_max(hor_r2_max.y, hor_r2_max.x);
        Point2i vert_s1_min(hor_s1_min.y, hor_s1_min.x);
        Point2i vert_s1_max(hor_s1_max.y, hor_s1_max.x);
        Point2i vert_s2_min(hor_s2_min.y, hor_s2_min.x);
        Point2i vert_s2_max(hor_s2_max.y, hor_s2_max.x);

        // calculate integral image
        Mat im_grid_integral;
        integral(im_grid_thick, im_grid_integral, CV_32S);

        const int SEARCH_RADIUS = 15;   // from ideal center
        for (int i = 0; i < 9; i++) {
            // find ideal center of cell in row i, col j
            float ideal_center_y = round((i + 0.5) / 9 * GRID_DIM) + GRID_PAD;
            for (int j = 0; j < 9; j++) {
                // find ideal center of cell in row i, col j
                float ideal_center_x = round((j + 0.5) / 9 * GRID_DIM) + GRID_PAD;

                /*
                // search left to right with template of two vertical bars
                Point2i LR_best = twoRectsMatch(im_grid_integral,
                    vert_r1_min, vert_r1_max, vert_r2_min, vert_r2_max,
                    Point2i(ideal_center_x - SEARCH_RADIUS, ideal_center_y),
                    Point2i(ideal_center_x + SEARCH_RADIUS + 1, ideal_center_y + 1)
                    );

                // search top to bottom with template of two horizontal bars
                Point2i LRTB_best = twoRectsMatch(im_grid_integral,
                    hor_r1_min, hor_r1_max, hor_r2_min, hor_r2_max,
                    Point2i(LR_best.x, ideal_center_y - SEARCH_RADIUS),
                    Point2i(LR_best.x + 1, ideal_center_y + SEARCH_RADIUS + 1)
                    );
                */

                // search top to bottom with template of two horizontal bars
                Point2i TB_best = twoRectsMatch(im_grid_integral,
                    hor_r1_min, hor_r1_max, hor_r2_min, hor_r2_max,
                    hor_s1_min, hor_s1_max, hor_s2_min, hor_s2_max,
                    Point2i(ideal_center_x, ideal_center_y - SEARCH_RADIUS),
                    Point2i(ideal_center_x + 1, ideal_center_y + SEARCH_RADIUS + 1)
                    );
                // search left to right with template of two vertical bars
                Point2i LRTB_best = twoRectsMatch(im_grid_integral,
                    vert_r1_min, vert_r1_max, vert_r2_min, vert_r2_max,
                    vert_s1_min, vert_s1_max, vert_s2_min, vert_s2_max,
                    Point2i(ideal_center_x - SEARCH_RADIUS, TB_best.y),
                    Point2i(ideal_center_x + SEARCH_RADIUS + 1, TB_best.y + 1)
                    );
                // search top to bottom again with template of two horizontal bars
                Point2i LRTB_best2 = twoRectsMatch(im_grid_integral,
                    hor_r1_min, hor_r1_max, hor_r2_min, hor_r2_max,
                    hor_s1_min, hor_s1_max, hor_s2_min, hor_s2_max,
                    Point2i(LRTB_best.x, LRTB_best.y - SEARCH_RADIUS),
                    Point2i(LRTB_best.x + 1, LRTB_best.y + SEARCH_RADIUS + 1)
                    );
                cellCenters[9 * i + j] = LRTB_best2;
            }
        }
    }
    B = getTickCount();
    cout << "cell centers: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;


    // draw centers
    Mat im_cell_centers;
    cvtColor(im_grid_thick, im_cell_centers, CV_GRAY2RGB);
    for (int i = 0; i < cellCenters.size(); i++) {
        circle(im_cell_centers, cellCenters[i], 3, Scalar(0, 255, 0), 2, 8, 0);
    }
    namedWindow("cell centers", CV_WINDOW_AUTOSIZE);
    imshow("cell centers", im_cell_centers);




    // extract numbers from grid
    vector<Mat> numbers_bin(cellCenters.size());
    
    const int CELLRECT_HALF_DIM = 20;
    const int CELLRECT_DIM = 2 * CELLRECT_HALF_DIM + 1;
    const int CELLRECT_PIXELS = CELLRECT_DIM * CELLRECT_DIM;
    const int NUMRECT_DIM = 50;

    Point2i cellRectMin(-CELLRECT_HALF_DIM, -CELLRECT_HALF_DIM);
    Point2i cellRectMax(CELLRECT_HALF_DIM + 1, CELLRECT_HALF_DIM + 1);
    Rect cellRect(cellRectMin, cellRectMax);
    for (int i = 0; i < cellCenters.size(); i++) {
        // crop out a square around the cell center to get the number
        Mat im_num = im_grid(cellRect + cellCenters[i]);

        // if square has too few white pixels, it's assumed blank
        if (countNonZero(im_num) < CELLRECT_PIXELS / 16) {
            continue;
        }
        
        // morphological close ????

        // find bounds of largest blob in the square
        int maxArea = 0;
        Point maxBlobPoint;
        for (int y = 0; y < CELLRECT_DIM; y++) {
            uchar *row = im_num.ptr(y);
            for (int x = 0; x < CELLRECT_DIM; x++) {
                if (row[x] > 128) {
                    int area = floodFill(im_num, Point(x, y), Scalar(128));
                    if (area > maxArea) {
                        maxArea = area;
                        maxBlobPoint = Point(x, y);
                    }
                }
            }
        }
        Rect blobRect;
        floodFill(im_num, maxBlobPoint, Scalar(255), &blobRect);
        threshold(im_num, im_num, 254, 255, CV_THRESH_BINARY);

        // extract largest blob square bounding box and scale to a fixed size
        int widthDiff = CELLRECT_DIM - blobRect.height;
        Point2i topLeft(widthDiff / 2, blobRect.tl().y);
        Point2i bottomRight(topLeft.x + blobRect.height, blobRect.br().y);

        Mat im_num_cropped_scaled;
        resize(im_num(Rect(topLeft, bottomRight)), im_num_cropped_scaled, Size(NUMRECT_DIM, NUMRECT_DIM));

        numbers_bin[i] = im_num_cropped_scaled;
    }
    
    namedWindow("cell", CV_WINDOW_AUTOSIZE);
    imshow("cell", numbers_bin[36]);







    D = getTickCount();
    cout << endl << "TOTAL: " << ((D - C) / TICK_FREQ_MS) << " ms" << endl;

    waitKey();

    return 0;
}
//}



int getRectSum(const Mat& integral, const Point2i& min, const Point2i& max) {
    const int* min_row = integral.ptr<int>(min.y);
    const int* max_row = integral.ptr<int>(max.y);
    return (max_row[max.x] - max_row[min.x] - min_row[max.x] + min_row[min.x]);
}

// [r1_min, r1_max) are bounds of first rect relative to the center of this template
// [r2_min, r2_max) are bounds of the second rect
// [s1_min, s1_max), [s1_min, s1_max) are bounds of rects to whose sum is subtracted
// instead of added
// [center_min, center_max) are the bounds of the rect in which the center can
// move around to find a max match.
// centers that gave the max template sum value are recorded

Point2i twoRectsMatch(const Mat& integral,
    const Point2i& r1_min, const Point2i& r1_max,
    const Point2i& r2_min, const Point2i& r2_max,
    const Point2i& s1_min, const Point2i& s1_max,
    const Point2i& s2_min, const Point2i& s2_max,
    const Point2i& center_min, const Point2i& center_max) {

    int maxResponse = -2000000000;
    Point2i maxMatchesSum(0);
    int numMaxMatches = 0;
    for (int x = center_min.x; x < center_max.x; x++) {
        for (int y = center_min.y; y < center_max.y; y++) {
            
            Point2i center(x, y);
            
            int r1_sum = getRectSum(integral, center + r1_min, center + r1_max);
            int r2_sum = getRectSum(integral, center + r2_min, center + r2_max);
            int s1_sum = getRectSum(integral, center + s1_min, center + s1_max);
            int s2_sum = getRectSum(integral, center + s2_min, center + s2_max);

            int response = min(r1_sum - s1_sum, r2_sum - s2_sum);

            if (response > maxResponse) {
                maxResponse = response;
                maxMatchesSum = center;
                numMaxMatches = 1;
            } else if (response == maxResponse) {
                maxMatchesSum += center;
                numMaxMatches++;
            }
        }
    }
    return maxMatchesSum * (1.0 / numMaxMatches);
}


