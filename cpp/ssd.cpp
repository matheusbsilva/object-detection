#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;


int main() {
    int inpWidth = 300;
    int inpHeight = 300;
    float conf_threshold = 0.4;

    string modelcfg = "ssd/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
    string modelweights = "ssd/frozen_inference_graph.pb";

    Net net = readNetFromTensorflow(modelweights, modelcfg);

    VideoCapture cap;
    cap.open(2);

    Mat frame;

    for(;;) {
        Mat blob;
        Mat outs;

        cap.read(frame);
        resize(frame, frame, Size(inpWidth, inpHeight));

        blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

        net.setInput(blob);
        outs = net.forward("detection_out");

        int detections = outs.size[2];

        for(int i = 0; i < detections; i++) {

            int id_conf[4] = {0, 0, i, 2};

            float confidence = (float)outs.at<float>(id_conf);

            cout << "Conf: " << confidence << endl; 
            if(confidence > conf_threshold) {
                int id_xlb[4] = {0, 0, i, 3};
                int id_ylb[4] = {0, 0, i, 4};
                int id_xrt[4] = {0, 0, i, 5};
                int id_yrt[4] = {0, 0, i, 6};

                int xLeftBottom = outs.at<float>(id_xlb) * frame.cols;
                int yLeftBottom = outs.at<float>(id_ylb) * frame.rows;
                int xRightTop = outs.at<float>(id_xrt) * frame.cols;
                int yRightTop = outs.at<float>(id_yrt) * frame.rows;

                cout << xLeftBottom << endl;

                rectangle(frame, Point(xLeftBottom, yLeftBottom), 
                          Point(xRightTop, yRightTop), Scalar(0, 255, 0));

            }

        }

        imshow("Objects", frame);
        if(waitKey(1) == 27)
            break;
    }

    return 0;
}
