#include <cstdlib>
#include <iostream>

//OpenCV
#include "opencv2/core.hpp"
#include <opencv2/imgcodecs.hpp>

//darknet with GPU
#define OPENCV
#define GPU
#include "darknet.h"
#include "image_opencv.h"

int main(int argc, char** argv) 
{

    std::string model_file = "/data/ml/train/darknet53-train/cfg/darknet53.cfg";
    std::string weights_file = "/data/ml/train/darknet53-train/weights/darknet53_last.weights";

    //load network & weights
    network* net = load_network((char*)model_file.c_str(), (char*)weights_file.c_str(), 1);

    //batch normalization
    fuse_conv_batchnorm(*net);
    calculate_binary_weights(*net);
                

    //load img
    std::string image_file = "/data/ml/train/darknet53-train/dataset/foo.png";
    cv::Mat img_mat =  cv::imread(image_file);
    
    
    //load Mat into darknet image...
    auto image_ptr = mat_to_image_cv((mat_cv*) &img_mat);
    
    //do precdiction
    float* ret = network_predict_image(net, image_ptr);
    
    float class1 = (ret[0] > 0.01f) ? ret[0] : 0 ;
    float class_background = (ret[1] > 0.01f) ? ret[1] : 0 ;
    
    std::cout << "Class1:" << class1 << ", Background:" << class_background << std::endl;
    
    
    
    return 0;
}

