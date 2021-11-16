#include "vision/choi_cv.cpp"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"


// #include "g2o/types/sba/types_six_dof_expmap.h"

// G2O_USE_TYPE_GROUP(slam2d);
// G2O_USE_TYPE_GROUP(slam3d);

int main(int argc, char **argv)
{
  unsigned long frame_number = 10;

  ros::init(argc, argv, "test");
  ros::NodeHandle nh;

  sensor_msgs::Image prev_image;
  sensor_msgs::Image curr_image;

  sensor_msgs::Image left_image;
  sensor_msgs::Image right_image;


  sensor_msgs::Image match_image;
  cv_bridge::CvImage cv_bridge;

  ros::Publisher img_prev_pub = nh.advertise<sensor_msgs::Image>("prev_image",1000);
  ros::Publisher img_curr_pub = nh.advertise<sensor_msgs::Image>("curr_image",1000);

  ros::Publisher img_left_pub = nh.advertise<sensor_msgs::Image>("left_image",1000);
  ros::Publisher img_right_pub = nh.advertise<sensor_msgs::Image>("right_image",1000);

  ros::Publisher img_match_pub = nh.advertise<sensor_msgs::Image>("match_image",1000);
  ros::Publisher camera_pose = nh.advertise<visualization_msgs::MarkerArray>("camera_pose",1000);

  //camera information//////////////////////////////
  double base_line_meter = 0.53715;
  double fx = 7.215377 * 100;
  double fy = fx;
  double cx = 6.095593*100;
  double cy = 1.728540*100;
  double cam_pix_size = 4.65 * 0.000001;

  Eigen::Vector2d focal_length(fx,fy); // pixels
  Eigen::Vector2d principal_point(cx,cy);

  choi::frame *frm;
  frm = new choi::frame[10];

  std::string img_l_path0 = "/home/csw/cv/KITTI/039/image_00/data/0000000002.png";
  std::string img_r_path0 = "/home/csw/cv/KITTI/039/image_01/data/0000000002.png";

  cv::Mat img_left0 = cv::imread(img_l_path0, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right0 = cv::imread(img_r_path0, CV_LOAD_IMAGE_GRAYSCALE);
  frm[0].img_l = img_left0;
  frm[0].img_r = img_right0;
  frm[0].feature_extract();
  frm[0].feature_match();
  frm[0].sort_match(); // need
  frm[0].triangulation();

  std::string img_l_path1 = "/home/csw/cv/KITTI/039/image_00/data/0000000003.png";
  std::string img_r_path1 = "/home/csw/cv/KITTI/039/image_01/data/0000000003.png";

  cv::Mat img_left1 = cv::imread(img_l_path1, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_right1 = cv::imread(img_r_path1, CV_LOAD_IMAGE_GRAYSCALE);
  frm[1].img_l = img_left1;
  frm[1].img_r = img_right1;
  frm[1].feature_extract();
  frm[1].feature_match();
  frm[1].sort_match(); // need
  frm[1].triangulation();

  cv::Mat prev = cv::imread(img_l_path0,CV_LOAD_IMAGE_COLOR);
  cv::Mat current = cv::imread(img_l_path1,CV_LOAD_IMAGE_COLOR);

  cv::Mat left = cv::imread(img_l_path0,CV_LOAD_IMAGE_COLOR);
  cv::Mat right = cv::imread(img_r_path0,CV_LOAD_IMAGE_COLOR);

  //draw left right matching
  for(int j = 0; j < 20; j++){
    cv::circle(left, cv::Point(static_cast<int>(frm[0].kp_l[frm[0].matches[j].queryIdx].pt.x),
        static_cast<int>(frm[0].kp_l[frm[0].matches[j].queryIdx].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);
    cv::circle(right, cv::Point(static_cast<int>(frm[0].kp_r[frm[0].matches[j].trainIdx].pt.x),
        static_cast<int>(frm[0].kp_r[frm[0].matches[j].trainIdx].pt.y)), 5, cv::Scalar(255,0,255), 2, 4, 0);

    double x = frm[0].coordinate_meter[frm[0].matches[j].queryIdx].x;
    double y = frm[0].coordinate_meter[frm[0].matches[j].queryIdx].y;
    double z = frm[0].coordinate_meter[frm[0].matches[j].queryIdx].z;

    //ROS_INFO("Tri   :  x : %lf, y : %lf, z : %lf, ",x,y,z);

  }

  //pose esimation using solvepnp
  std::vector<cv::Point3f> objectPoints;
  std::vector<cv::Point2f> imagePoints;

  std::vector<cv::DMatch> matches_frame; //matching container
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING); //need improve BF this is not efficient
  matcher->match(frm[0].des_l,frm[1].des_l,matches_frame);// match feature prev_frame_left with curr_frame_left
  std::sort(matches_frame.begin(),matches_frame.end());

  cv::Point3f point3;
  cv::Point2f point2;
  int point_num = 0;

  //add objectPoints and imagePoints
  for(unsigned long k = 0; k < matches_frame.size(); k++){
    unsigned long prev_match_idx = static_cast<unsigned long>(matches_frame[k].queryIdx);
    unsigned long curr_match_idx = static_cast<unsigned long>(matches_frame[k].trainIdx);

    if(frm[0].coordinate_meter[prev_match_idx].x != 0.0
       && frm[0].coordinate_meter[prev_match_idx].z < 30.0){

      if(std::abs(frm[0].kp_l[prev_match_idx].pt.x -
                  frm[1].kp_l[curr_match_idx].pt.x) > 100){
        continue;
      }

      point3.x = static_cast<float>(frm[0].coordinate_meter[prev_match_idx].x/cam_pix_size);
      point3.y = static_cast<float>(frm[0].coordinate_meter[prev_match_idx].y/cam_pix_size);
      point3.z = static_cast<float>(frm[0].coordinate_meter[prev_match_idx].z/cam_pix_size);

      //ROS_INFO("x=%f, y=%f, z=%f",point3.x, point3.y, point3.z);
      ROS_INFO("dist : %f",matches_frame[k].distance);
      ROS_INFO("x=%f, y=%f, z=%f, num:k %d",point3.x*cam_pix_size, point3.y*cam_pix_size, point3.z*cam_pix_size, point_num);

      point2.x = static_cast<float>(frm[1].kp_l[curr_match_idx].pt.x);
      point2.y = static_cast<float>(frm[1].kp_l[curr_match_idx].pt.y);
      //ROS_INFO("x=%f, y=%f",point2.x, point2.y);

      //ROS_INFO("X1 = %f, X2 = %f , k = %d",frm[0].kp_l[static_cast<unsigned long>(matches_frame[k].queryIdx)].pt.x,
      //    frm[1].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.x,static_cast<int>(k));

      cv::circle(prev, cv::Point(static_cast<int>(frm[0].kp_l[static_cast<unsigned long>(matches_frame[k].queryIdx)].pt.x),
          static_cast<int>(frm[0].kp_l[static_cast<unsigned long>(matches_frame[k].queryIdx)].pt.y)), 5, cv::Scalar((k* 58 + 39) % 255,(k* 27 + 39) % 255,(k* 60 + 50) % 255), 2, 4, 0);
      cv::circle(current, cv::Point(static_cast<int>(frm[1].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.x),
          static_cast<int>(frm[1].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.y)), 5, cv::Scalar((k* 58 + 39) % 255,(k* 27 + 39) % 255,(k* 60 + 50) % 255), 2, 4, 0);

      objectPoints.push_back(point3); //add object point
      imagePoints.push_back(point2);  //add image  point

      point_num++;  //count num of points
    }
    if(point_num > 20){// later use global
      break;
    }
  }


  double intrinsic[] = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};  //camera intrinsic parameter
  cv::Mat cameraMat(3,3, CV_64FC1, intrinsic); //camera matrix

  double distortion[] = {0, 0, 0, 0}; //data has zero distortion because already recitified
  cv::Mat distCoeffs(4, 1, CV_64FC1, distortion);

  //estimate camera pose
  cv::Mat rvec, tvec; //rotaion, translation matrix

  cv::solvePnP(objectPoints,imagePoints,cameraMat,distCoeffs,rvec,tvec);
  cv::Mat R_o;
  cv::Rodrigues(rvec,R_o);
  cv::Mat R_c = R_o.inv();

  cv::Mat P_c = -R_c*tvec;
  double *p_c = (double*)P_c.data;

  ROS_INFO("x=%lf, y=%lf, z=%lf, frame", p_c[0]*cam_pix_size, p_c[1]*cam_pix_size, p_c[2]*cam_pix_size);


  ros::Rate loop_rate(4);
  std_msgs::Header header;
  while (ros::ok())
  {

    std_msgs::String msg;
    msg.data = "hello world";

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, prev);
    cv_bridge.toImageMsg(prev_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, current);
    cv_bridge.toImageMsg(curr_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, left);
    cv_bridge.toImageMsg(left_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, right);
    cv_bridge.toImageMsg(right_image);

    //cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, dst);
    //cv_bridge.toImageMsg(match_image);

    prev_image.header.stamp = ros::Time::now();
    prev_image.header.frame_id = "camera";
    curr_image.header.stamp = ros::Time::now();
    curr_image.header.frame_id = "camera";

    left_image.header.stamp = ros::Time::now();
    left_image.header.frame_id = "camera";
    right_image.header.stamp = ros::Time::now();
    right_image.header.frame_id = "camera";


    match_image.header.stamp = ros::Time::now();
    match_image.header.frame_id = "camera";



    img_prev_pub.publish(prev_image);
    img_curr_pub.publish(curr_image);
    img_left_pub.publish(left_image);
    img_right_pub.publish(right_image);
    //img_match_pub.publish(match_image);



    ros::spinOnce();
    loop_rate.sleep();
  }

  delete[] frm;
  return 0;
}
