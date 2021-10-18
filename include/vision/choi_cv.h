#ifndef CHOI_CV_H
#define CHOI_CV_H
//
#include "ros/ros.h"
//
#include "std_msgs/String.h"
#include <std_msgs/UInt8MultiArray.h>
#include <sensor_msgs/Image.h>
#include<sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
//
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
//
#include <vector>
namespace choi {
  bool cmp_resp_desc(const cv::KeyPoint &a, const cv::KeyPoint &b); //compare response

  void extract(const cv::Mat& img, cv::Mat& des, std::vector<cv::KeyPoint>& kp);


  //object for every frame
  class frame{

  public:
  //variable
    //stereo image
    cv::Mat img_l;
    cv::Mat img_r;

    //feature
    std::vector<cv::KeyPoint> kp_l, kp_r; //keypoint container
    cv::Mat des_l, des_r; //keypoint desciptor

    std::vector<cv::DMatch> matches; //matching container
    std::vector<cv::DMatch> good_matches; //Top n matches //query => left kp idx, train => right kp idx

  //function
    frame(cv::Mat img1, cv::Mat img2); //img1 => img_l, img2 => img_r
    //feature function
    void feature_extract();
    void feature_match(); //matching left, right feature
    void sort_match(); //sort matches


    //draw
    void draw_feature(cv::Mat &img1, cv::Mat &img2);
    void draw_feature_onframe();
    void draw_match(cv::Mat &dst);
    void draw_byFeatureIdx_onframe(int idx_l, int idx_r);

  };
}



#endif // CHOI_CV_H
