#pragma once
#include "vision/choi_cv.h"


namespace choi {

  bool cmp_resp_desc(const cv::KeyPoint &a, const cv::KeyPoint &b) {
    if (a.response > b.response) return true;
    else return false;
  }

  //feature extract from input image using ORB
  void extract(const cv::Mat& img, cv::Mat& des, std::vector<cv::KeyPoint>& kp) {
      const static auto& orb = cv::ORB::create();
     orb->detectAndCompute(img, cv::noArray(), kp, des);
  }

  //feature sort rule
  void sort_kp_desc(std::vector<cv::KeyPoint>& kp){
    std::sort(kp.begin(), kp.end(),cmp_resp_desc);
  }


//frame class//////////////////////////////////////////////////////////////////
  frame::frame(cv::Mat img1, cv::Mat img2){
    img_l = img1; img_r = img2;
  }

  //Feature extract from left and right images using ORB
  void frame::feature_extract(){
    const static auto& orb = cv::ORB::create();
    orb->detectAndCompute(img_l, cv::noArray(), kp_l, des_l);
    orb->detectAndCompute(img_r, cv::noArray(), kp_r, des_r);
  }

  //left and right image feature matching
  void frame::feature_match(){
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING,true); //need improve BF not efficient
    matcher->match(des_l,des_r,matches);
  }

  //Sort matches Descending
  void frame::sort_match(){
    std::sort(matches.begin(),matches.end());
  }

  //Calculate 3D coordinate using Triangulation
  void frame::triangulation(double fx, double cx, double cy){//defalue kitti 2011-09-26
    //camera information  //after change to define or member
    double base_line_meter = 0.53715;
    double cam_pix_size = 4.65 * 0.000001;
    double fy = fx;

    double base_line_pix = base_line_meter / cam_pix_size;

    int left_idx, right_idx;
    float left_x, left_y, right_x, right_y;

    for(int i = 0; i < matches.size(); i++){

      left_idx  = matches[i].queryIdx;
      right_idx = matches[i].trainIdx;

      left_x  = kp_l[static_cast<unsigned long>(left_idx)].pt.x;
      left_y  = kp_l[static_cast<unsigned long>(left_idx)].pt.y;
      right_x = kp_r[static_cast<unsigned long>(right_idx)].pt.x;
      right_y = kp_r[static_cast<unsigned long>(right_idx)].pt.y;

      //float right_y = kp_r[static_cast<unsigned long>(right_idx)].pt.y;

      //pixel unit
      //Triangulation
      if(std::abs(left_x - right_x) > 40 || left_x - right_x < 2) continue; // error reject
      //if(std::abs(left_y - right_y) > 20) continue;

      double z = (base_line_pix * fx) / (static_cast<double>(left_x) - static_cast<double>(right_x));
      double x = (static_cast<double>(left_x) - cx)  * z / fx;
      double y = (static_cast<double>(left_y) - cy) * z / fx;

      //meter unit
      coordinate_meter[left_idx].z = z*cam_pix_size;
      coordinate_meter[left_idx].x = x*cam_pix_size;
      coordinate_meter[left_idx].y = y*cam_pix_size;

      //ROS_INFO("dp : %f", (left_x - right_x));
    }
  }


  //Draw featrue on input image
  void frame::draw_feature_both(cv::Mat &img1, cv::Mat &img2){
    img1 = img_l;
    img2 = img_r;
    for(std::vector<cv::KeyPoint>::size_type  i = 0; i < kp_l.size(); i++){
      cv::circle(img1, cv::Point(static_cast<int>(kp_l[i].pt.x),static_cast<int>(kp_l[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      cv::circle(img2, cv::Point(static_cast<int>(kp_r[i].pt.x),static_cast<int>(kp_r[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      //ROS_INFO("%d : %lf",static_cast<int>(i), static_cast<double>(kp[i].response));
    }
  }

  //Draw featrue on input image
  void frame::draw_feature(cv::Mat &img1){
    //img1 = img_l;
    for(std::vector<cv::KeyPoint>::size_type  i = 0; i < kp_l.size(); i++){
      cv::circle(img1, cv::Point(static_cast<int>(kp_l[i].pt.x),static_cast<int>(kp_l[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
    }
  }

  //Draw feature on left, right image
  void frame::draw_feature_onframe(){
    for(std::vector<cv::KeyPoint>::size_type  i = 0; i < kp_l.size(); i++){
      cv::circle(img_l, cv::Point(static_cast<int>(kp_l[i].pt.x),static_cast<int>(kp_l[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      cv::circle(img_r, cv::Point(static_cast<int>(kp_r[i].pt.x),static_cast<int>(kp_r[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      //ROS_INFO("%d : %lf",static_cast<int>(i), static_cast<double>(kp[i].response));
    }
  }

  //Draw match on input image
  void frame::draw_match(cv::Mat &dst){
    cv::drawMatches(img_l,kp_l,img_r,kp_r,good_matches,dst,cv::Scalar::all(-1), cv::Scalar(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  }

  //Draw a feature using the idx of the matched feature
  void frame::draw_byFeatureIdx_onframe(int idx_l, int idx_r){
    cv::circle(img_l, cv::Point(static_cast<int>(kp_l[idx_l].pt.x),static_cast<int>(kp_l[idx_l].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
    cv::circle(img_r, cv::Point(static_cast<int>(kp_r[idx_r].pt.x),static_cast<int>(kp_r[idx_r].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
  }

}


