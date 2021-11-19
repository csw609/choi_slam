#ifndef CHOI_CV_H
#define CHOI_CV_H
// //////////
#include <iostream>
#include <vector>
// //////////
#include "ros/ros.h"
// //////////
#include "std_msgs/String.h"
#include <std_msgs/UInt8MultiArray.h>
#include <sensor_msgs/Image.h>
#include<sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/OccupancyGrid.h>
// //////////
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
// //////////
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
// #include "g2o/solvers/csparse/linear_solver_csparse.h"
// #include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
// #include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/factory.h"
#include "g2o/types/slam3d/vertex_se3.h"
#include "g2o/types/slam3d/edge_se3.h"

#include "g2o/types/icp/types_icp.h"
// //////////
//#define BASE_LINE_METER  0.54
//#define FOCAL_LENGTH_X_PIXEL  721.5377
//#define FOCAL_LENGTH_Y_PIXEL  721.5377

#define IMAGE_WIDTH 1242
#define IMAGE_HEIGHT 375

struct coordinate_3D{
  double x;
  double y;
  double z;
};

namespace choi {
  bool cmp_resp_desc(const cv::KeyPoint &a, const cv::KeyPoint &b); //compare response

  void extract(const cv::Mat& img, cv::Mat& des, std::vector<cv::KeyPoint>& kp);

//frame class/////////////////////////////////////////////////////////////////////////////////////
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

    std::vector<cv::DMatch> matches; //stereo feature matching container
    std::vector<cv::DMatch> matches_with_prev; //prev curr feature matching container
    std::vector<cv::DMatch> good_matches; //Top n matches //query => left kp idx, train => right kp idx
    coordinate_3D coordinate_meter[501];  //3D coordinate of feature in current camera frame idx follow kp_l



    //world coordinate
    Eigen::Vector3d trans_cam;
    Eigen::Quaterniond q_cam;
    Eigen::Isometry3d pose_c2w;

    //to prev frame
    Eigen::Isometry3d pose_prev;


    //function
    frame(cv::Mat img1, cv::Mat img2); //img1 => img_l, img2 => img_r
    frame(){};

    //feature function
    void feature_extract();
    void feature_match(); //matching left, right feature
    void sort_match(); //sort matches
    void triangulation(double fx = 7.215377 * 100, double cx = 6.095593*100, double cy = 1.728540*100); //calculate 3D coordinate

    //coordinate
    void init_pose(){pose_c2w = q_cam; pose_c2w.translation() = trans_cam;};

    //draw
    void draw_feature_both(cv::Mat &img1, cv::Mat &img2);
    void draw_feature(cv::Mat &img1);
    void draw_feature_onframe();
    void draw_match(cv::Mat &dst);
    void draw_byFeatureIdx_onframe(int idx_l, int idx_r);

  };

//frame class////////////////////////////////////////////////////////////////////////////////////
}



#endif // CHOI_CV_H
