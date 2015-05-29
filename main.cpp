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
    




    /*// dilate image to cover up small holes
    A = getTickCount();
    Mat dilate_SE;
    {
        int SE_radius = max((int)((1.0 / 256.0) * IM_SQRT_RES), 1);
        int SE_size = SE_radius * 2 + 1;
        cout << "dilate SE size " << SE_size << endl;
        dilate_SE = Mat::zeros(SE_size, SE_size, CV_8U);
        for (int y = 0; y < SE_size; y++) 
            dilate_SE.ptr(y)[SE_radius] = 1;
        uchar* row = dilate_SE.ptr(SE_radius);
        for (int x = 0; x < SE_size; x++)
            row[x] = 1;
        dilate(im_bin, im_bin, dilate_SE);
    }
    B = getTickCount();
    cout << "dilate: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;
    */

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
    B = getTickCount();
    cout << "find max blob: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    //namedWindow("max blob", WINDOW_AUTOSIZE);
    //imshow("max blob", im_binblob);



    
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
    

    /*
    // find lines in the grid
    
    A = getTickCount();
    vector<Vec2f> lines;
    {
        double rho = (1.0 / 512.0) * IM_SQRT_RES;   //1;
        double theta = 1.0 * (CV_PI / 180.0);
        int threshold = (int)(IM_SQRT_RES / 5.0);//100;
        HoughLines(im_edge, lines, rho, theta, threshold, 0, 0);
    }
    B = getTickCount();
    cout << "find lines: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    // draw lines
    Mat im_lines;
    im.copyTo(im_lines);
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(im_lines, pt1, pt2, Scalar(0, 255, 0), 1, CV_AA);
    }

    namedWindow("detected lines", WINDOW_AUTOSIZE);
    imshow("detected lines", im_lines);
    */
    /*
    A = getTickCount();
    vector<Vec4i> lines;
    double rho = (1.0 / 512.0) * IM_SQRT_RES;
    double theta = 1.0 * (CV_PI / 180.0);
    int threshold = 100;
    HoughLinesP(im_edge, lines, rho, theta, threshold);
    B = getTickCount();
    cout << "find lines: " << ((B - A) / TICK_FREQ_MS) << " ms" << endl;

    Mat im_lines;
    im.copyTo(im_lines);
    for (size_t i = 0; i < lines.size(); i++) {
    Vec4i l = lines[i];
    line(im_lines, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
    }
    namedWindow("detected lines", WINDOW_AUTOSIZE);
    imshow("detected lines", im_lines);
    */


    //if (lines.size() < 20)
    //    return -1;


    // find the four corners of the grid
    A = getTickCount();
    vector<Point2f> corners(4);
    /*{
        // change thetas from [0,pi) to [pi/4,5pi/4)
        // also find the max and min theta values
        float minTheta = 100.0;
        float maxTheta = -100.0;
        for (int i = 0; i < lines.size(); i++){
        float rho = lines[i][0];
        float theta = lines[i][1];
        if (theta < CV_PI / 4.0) {
        theta += CV_PI;
        rho = -rho;
        lines[i][0] = rho;
        lines[i][1] = theta;
        }
        if (theta < minTheta)
        minTheta = theta;
        if (theta > maxTheta)
        maxTheta = theta;
        }
        // partition the thetas with the avg of the min and max as the threshold
        vector<bool> thetaAbove(lines.size());
        float threshold = 0.5f * (minTheta + maxTheta);
        for (int i = 0; i < lines.size(); i++) {
        thetaAbove[i] = (lines[i][1] >= threshold);
        }
        // for each partition, find the lines with the min and max rhos
        float lowThetaMinRho = INFINITY;
        float lowThetaMaxRho = -INFINITY;
        int lowThetaMinMaxIndices[2];   // index 0 is min, 1 is max
        float highThetaMinRho = INFINITY;
        float highThetaMaxRho = -INFINITY;
        int highThetaMinMaxIndices[2];  // index 0 is min, 1 is max
        for (int i = 0; i < lines.size(); i++) {
        float rho = lines[i][0];
        if (!thetaAbove[i]) {
        if (rho < lowThetaMinRho) {
        lowThetaMinRho = rho;
        lowThetaMinMaxIndices[0] = i;
        }
        if (rho > lowThetaMaxRho) {
        lowThetaMaxRho = rho;
        lowThetaMinMaxIndices[1] = i;
        }
        } else {
        if (rho < highThetaMinRho) {
        highThetaMinRho = rho;
        highThetaMinMaxIndices[0] = i;
        }
        if (rho > highThetaMaxRho) {
        highThetaMaxRho = rho;
        highThetaMinMaxIndices[1] = i;
        }
        }
        }
        // find intersections for the following pairs of lines:
        // lowMin-highMin, lowMin-highMax, lowMax-highMin, lowMax-highMax
        for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
        Vec2f& line1 = lines[lowThetaMinMaxIndices[i]];
        Vec2f& line2 = lines[highThetaMinMaxIndices[j]];
        float rho1 = line1[0], theta1 = line1[1];
        float rho2 = line2[0], theta2 = line2[1];
        // solve Ax=b for x, the intersection point
        float a = cosf(theta1), b = sinf(theta1);
        float c = cosf(theta2), d = sinf(theta2);
        float inv_det = 1.0f / (a*d - b*c);
        float x = inv_det*(d*rho1 - b*rho2);
        float y = inv_det*(a*rho2 - c*rho1);
        corners[2 * i + j] = Point(x, y);
        }
        }

        }*/
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

    // erode lines??
    /*{
        int SE_length = ((int)((GRID_DIM / 9.0) * 0.5)) * 2 + 1;
        Mat SE = Mat::ones(1, SE_length, CV_8U);
        erode(im_grid, im_grid, SE);
    }*/

    /*// draw ideal 9x9 grid lines for visualization purposes
    Mat im_grid_with_lines;
    cvtColor(im_grid, im_grid_with_lines, CV_GRAY2RGB);
    for (int i = 1; i < 9; i++) {
        int d = round(GRID_DIM / 9.0 * i);
        Point pt1(0, d);
        Point pt2(GRID_DIM - 1, d);
        line(im_grid_with_lines, pt1, pt2, Scalar(0, 255, 0), 1, CV_AA);
        Point pt3(d, 0);
        Point pt4(d, GRID_DIM - 1);
        line(im_grid_with_lines, pt3, pt4, Scalar(0, 255, 0), 1, CV_AA);
    }
    namedWindow("grid", CV_WINDOW_AUTOSIZE);
    imshow("grid", im_grid_with_lines);
    */

    //imwrite("../grid.png", im_grid);
    




    


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

        // pad grid image
        //const int GRID_PAD = 30;
        //const int GRID_DIM_PAD = GRID_DIM + 2 * GRID_PAD;
        //Mat im_grid_border;
        //copyMakeBorder(im_grid_thick, im_grid_border, GRID_PAD, GRID_PAD, GRID_PAD, GRID_PAD,
        //    BORDER_CONSTANT, Scalar(0));

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


