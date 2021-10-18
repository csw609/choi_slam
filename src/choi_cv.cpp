#include "vision/choi_cv.h"
#include <iostream>

namespace choi {

  bool cmp_resp_desc(const cv::KeyPoint &a, const cv::KeyPoint &b) {
    if (a.response > b.response) return true;
    else return false;
  }

  //// 템플릿 및 카메라 프레임 특징점 추출
  void extract(const cv::Mat& img, cv::Mat& des, std::vector<cv::KeyPoint>& kp) {
      const static auto& orb = cv::ORB::create();
     orb->detectAndCompute(img, cv::noArray(), kp, des);
  }

  void sort_kp_desc(std::vector<cv::KeyPoint>& kp){
    std::sort(kp.begin(), kp.end(),cmp_resp_desc);
  }


  frame::frame(cv::Mat img1, cv::Mat img2){
    img_l = img1; img_r = img2;
  }

  void frame::feature_extract(){
    const static auto& orb = cv::ORB::create();
    orb->detectAndCompute(img_l, cv::noArray(), kp_l, des_l);
    orb->detectAndCompute(img_r, cv::noArray(), kp_r, des_r);
  }

  void frame::feature_match(){
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    matcher->match(des_l,des_r,matches);
  }

  void frame::sort_match(){
    std::sort(matches.begin(),matches.end());
    std::vector<cv::DMatch> good(matches.begin(), matches.begin() + 5);
    good_matches = good;

  }

  void frame::draw_feature(cv::Mat &img1, cv::Mat &img2){
    img1 = img_l;
    img2 = img_r;
    for(std::vector<cv::KeyPoint>::size_type  i = 0; i < kp_l.size(); i++){
      cv::circle(img1, cv::Point(static_cast<int>(kp_l[i].pt.x),static_cast<int>(kp_l[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      cv::circle(img2, cv::Point(static_cast<int>(kp_r[i].pt.x),static_cast<int>(kp_r[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      //ROS_INFO("%d : %lf",static_cast<int>(i), static_cast<double>(kp[i].response));
    }
  }

  void frame::draw_feature_onframe(){
    for(std::vector<cv::KeyPoint>::size_type  i = 0; i < kp_l.size(); i++){
      cv::circle(img_l, cv::Point(static_cast<int>(kp_l[i].pt.x),static_cast<int>(kp_l[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      cv::circle(img_r, cv::Point(static_cast<int>(kp_r[i].pt.x),static_cast<int>(kp_r[i].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
      //ROS_INFO("%d : %lf",static_cast<int>(i), static_cast<double>(kp[i].response));
    }
  }

  void frame::draw_match(cv::Mat &dst){
    cv::drawMatches(img_l,kp_l,img_r,kp_r,good_matches,dst,cv::Scalar::all(-1), cv::Scalar(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  }

  void frame::draw_byFeatureIdx_onframe(int idx_l, int idx_r){
    cv::circle(img_l, cv::Point(static_cast<int>(kp_l[idx_l].pt.x),static_cast<int>(kp_l[idx_l].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
    cv::circle(img_r, cv::Point(static_cast<int>(kp_r[idx_r].pt.x),static_cast<int>(kp_r[idx_r].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
  }

}


