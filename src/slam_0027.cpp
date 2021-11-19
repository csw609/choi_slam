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
  unsigned long frame_number = 1000;

  ros::init(argc, argv, "test");
  ros::NodeHandle nh;

  sensor_msgs::Image left_image;
  sensor_msgs::Image right_image;
  sensor_msgs::Image match_image;
  cv_bridge::CvImage cv_bridge;

  //rviz visualization
  visualization_msgs::Marker marker_cam_tmp;
  visualization_msgs::Marker marker_feature_tmp;

  visualization_msgs::MarkerArray marker_cam;
  visualization_msgs::MarkerArray marker_feature;

  geometry_msgs::Pose marker_cam_pose;
  geometry_msgs::Pose marker_feature_pose;

  nav_msgs::OccupancyGrid map;

  map.header.frame_id = "map";
  map.header.seq = 1;
  map.header.stamp.sec = 0;
  map.header.stamp.nsec = 0;
  map.info.width = 250;
  map.info.height = 500;
  map.info.origin.position.x = -125;
  map.info.origin.position.y = 0;
  map.info.origin.orientation.w = 1.0;
  map.info.origin.orientation.x = 0.0;
  map.info.origin.orientation.y = 0.0;
  map.info.origin.orientation.z = 0.0;
  map.info.resolution = 1.0;
  map.data.resize(map.info.width * map.info.height);
  for(unsigned long i = 0; i < map.info.width*map.info.height; i++){
    map.data[i] = -1;
  }


  marker_cam_pose.orientation.w = 1.0;
  marker_cam_pose.orientation.x = 0.0;
  marker_cam_pose.orientation.y = 0.0;
  marker_cam_pose.orientation.z = 0.0;

  marker_feature_pose.orientation.w = 1.0;
  marker_feature_pose.orientation.x = 0.0;
  marker_feature_pose.orientation.y = 0.0;
  marker_feature_pose.orientation.z = 0.0;

  marker_cam_tmp.header.frame_id = "camera";
  marker_cam_tmp.header.seq = 1;
  marker_cam_tmp.header.stamp.sec = 0;
  marker_cam_tmp.header.stamp.nsec = 0;
  marker_cam_tmp.type = 2; //sphere
  marker_cam_tmp.scale.x = 1.25;
  marker_cam_tmp.scale.y = 1.25;
  marker_cam_tmp.scale.z = 1.25;
  marker_cam_tmp.color.r = 1.0;
  marker_cam_tmp.color.g = 0.0;
  marker_cam_tmp.color.b = 0.0;
  marker_cam_tmp.color.a = 0.5;
  marker_cam_tmp.id = 0;

  marker_feature_tmp.header.frame_id = "camera";
  marker_feature_tmp.header.seq = 1;
  marker_feature_tmp.header.stamp.sec = 0;
  marker_feature_tmp.header.stamp.nsec = 0;
  marker_feature_tmp.type = 2; //sphere
  marker_feature_tmp.scale.x = 3.6;
  marker_feature_tmp.scale.y = 3.6;
  marker_feature_tmp.scale.z = 3.6;
  marker_feature_tmp.color.r = 0.0;
  marker_feature_tmp.color.g = 0.0;
  marker_feature_tmp.color.b = 1.0;
  marker_feature_tmp.color.a = 0.5;
  marker_feature_tmp.id = 0;

  ros::Publisher img_l_pub = nh.advertise<sensor_msgs::Image>("left_image",1000);
  ros::Publisher img_r_pub = nh.advertise<sensor_msgs::Image>("right_image",1000);
  ros::Publisher img_match_pub = nh.advertise<sensor_msgs::Image>("match_image",1000);

  ros::Publisher camera_pose_pub = nh.advertise<visualization_msgs::MarkerArray>("camera_pose",1000);
  ros::Publisher feature_pose_pub = nh.advertise<visualization_msgs::MarkerArray>("feature_pose",1000);

  ros::Publisher map_pub = nh.advertise<nav_msgs::OccupancyGrid>("map",1000);

  //camera information//////////////////////////////
  double fx = 7.18856 * 100;
  double fy = fx;
  double cx = 6.071928*100;
  double cy = 1.852157*100;
  double cam_pix_size = 4.65 * 0.000001;

  Eigen::Vector2d focal_length(fx,fy); // pixels
  Eigen::Vector2d principal_point(cx,cy);

  choi::frame *frm;
  frm = new choi::frame[frame_number];

  //pose2_vec
  ros::Rate loop_rate(20);
  std_msgs::Header header;
  unsigned long j = 0;
  while (ros::ok())
  {

    //ROS_INFO("debug %d", j);
    std::string img_l_path;
    std::string img_r_path;
    if(j < 10){
      img_l_path = "/home/csw/cv/pro/00.txt.d/camera_left.image_raw_0000000" + std::to_string(j) +".pgm";
      img_r_path = "/home/csw/cv/pro/00.txt.d/camera_right.image_raw_0000000" + std::to_string(j) +".pgm";
    }
    else if(j < 100){
      img_l_path = "/home/csw/cv/pro/00.txt.d/camera_left.image_raw_000000" + std::to_string(j) +".pgm";
      img_r_path = "/home/csw/cv/pro/00.txt.d/camera_right.image_raw_000000" + std::to_string(j) +".pgm";
    }
    else if(j < 1000){
      img_l_path = "/home/csw/cv/pro/00.txt.d/camera_left.image_raw_00000" + std::to_string(j) +".pgm";
      img_r_path = "/home/csw/cv/pro/00.txt.d/camera_right.image_raw_00000" + std::to_string(j) +".pgm";
    }
    else if(j < 10000){
      img_l_path = "/home/csw/cv/pro/00.txt.d/camera_left.image_raw_0000" + std::to_string(j) +".pgm";
      img_r_path = "/home/csw/cv/pro/00.txt.d/camera_right.image_raw_0000" + std::to_string(j) +".pgm";
    }

    cv::Mat img_left = cv::imread(img_l_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_right = cv::imread(img_r_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_show = cv::imread(img_l_path,CV_LOAD_IMAGE_COLOR);

    frm[j].img_l = img_left;
    frm[j].img_r = img_right;
    frm[j].feature_extract();
    frm[j].feature_match();
    frm[j].sort_match(); // need
    frm[j].triangulation(fx, cx, cy);

    if(j == 0){
      frm[0].trans_cam = Eigen::Vector3d(0,0,0);
      frm[0].q_cam     = Eigen::Quaterniond::Identity();
      frm[0].init_pose();
    }
    else if(j > 0) {
      //pose esimation using solvepnp
      std::vector<cv::Point3f> objectPoints;
      std::vector<cv::Point2f> imagePoints;

      std::vector<cv::DMatch> matches_frame; //matching container

      cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING,true); //need improve BF this is not efficient
      matcher->match(frm[j-1].des_l,frm[j].des_l,matches_frame);// match feature prev_frame_left with curr_frame_left
      std::sort(matches_frame.begin(),matches_frame.end());

      frm[j].matches_with_prev = matches_frame;

      cv::Point3f point3;
      cv::Point2f point2;
      int point_num = 0;
      //add objectPoints and imagePoints
      for(unsigned long k = 0; k < matches_frame.size(); k++){
        double x1 = frm[j-1].coordinate_meter[static_cast<unsigned long>(matches_frame[k].queryIdx)].x;
        double z1 = frm[j-1].coordinate_meter[static_cast<unsigned long>(matches_frame[k].queryIdx)].z;
        if(x1 != 0.0 && z1 < 30.0){
          if(std::abs(frm[j-1].kp_l[static_cast<unsigned long>(matches_frame[k].queryIdx)].pt.x -
                      frm[j].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.x) > 100) continue;

          point3.x = static_cast<float>(frm[j-1].coordinate_meter[static_cast<unsigned long>(matches_frame[k].queryIdx)].x/cam_pix_size);
          point3.y = static_cast<float>(frm[j-1].coordinate_meter[static_cast<unsigned long>(matches_frame[k].queryIdx)].y/cam_pix_size);
          point3.z = static_cast<float>(frm[j-1].coordinate_meter[static_cast<unsigned long>(matches_frame[k].queryIdx)].z/cam_pix_size);
          //ROS_INFO("x=%f, y=%f, z=%f",point3.x, point3.y, point3.z);
          //ROS_INFO("x=%f, y=%f, z=%f",point3.x*cam_pix_size, point3.y*cam_pix_size, point3.z*cam_pix_size);

          point2.x = static_cast<float>(frm[j].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.x);
          point2.y = static_cast<float>(frm[j].kp_l[static_cast<unsigned long>(matches_frame[k].trainIdx)].pt.y);
          //ROS_INFO("x=%f, y=%f",point2.x, point2.y);

          objectPoints.push_back(point3); //add object point
          imagePoints.push_back(point2);  //add image  point

          point_num++;  //count num of points
        }
        if(point_num > 50){// later use global
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
      Eigen::Vector3d prev2cur_trans;
      prev2cur_trans.x() = p_c[0]*cam_pix_size;
      prev2cur_trans.y() = p_c[1]*cam_pix_size;
      prev2cur_trans.z() = p_c[2]*cam_pix_size;
      Eigen::Matrix3d prev2cur_rot;

      cv::cv2eigen(R_c,prev2cur_rot);
      //ROS_INFO("x=%lf, y=%lf, z=%lf ,  frame%d", p_c[0]*cam_pix_size, p_c[1]*cam_pix_size, p_c[2]*cam_pix_size, static_cast<int>(j));

      //if reliable
      if(std::abs(prev2cur_trans.x()) < 1.0 && std::abs(prev2cur_trans.y()) < 0.35 && std::abs(prev2cur_trans.z()) < 1.0){
        frm[j].pose_prev = prev2cur_rot;
        frm[j].pose_prev.translation() = prev2cur_trans;
      }
      else{ //constant velocity motion model
        frm[j].pose_prev = frm[j-1].pose_prev;
      }
      ROS_INFO("filtered x=%lf, y=%lf, z=%lf ,  frame%d", frm[j].pose_prev.translation().x(), frm[j].pose_prev.translation().y(),
               frm[j].pose_prev.translation().z(), static_cast<int>(j));

      frm[j].pose_c2w = frm[j-1].pose_c2w * frm[j].pose_prev;

      ROS_INFO("world x=%lf, y=%lf, z=%lf ,  frame%d", frm[j].pose_c2w.translation().x(), frm[j].pose_c2w.translation().y(),
               frm[j].pose_c2w.translation().z(), static_cast<int>(j));
    }

    std_msgs::String msg;
    msg.data = "hello world";
    //frm[j].draw_feature_onframe();
    frm[j].draw_feature(img_show);
    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, img_show);
    cv_bridge.toImageMsg(left_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm[j].img_l);
    cv_bridge.toImageMsg(right_image);

    cv::Mat dst;
    frm[j].draw_match(dst);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, dst);
    cv_bridge.toImageMsg(match_image);

    left_image.header.stamp = ros::Time::now();
    left_image.header.frame_id = "camera";
    right_image.header.stamp = ros::Time::now();
    right_image.header.frame_id = "camera";
    match_image.header.stamp = ros::Time::now();
    match_image.header.frame_id = "camera";

    img_l_pub.publish(left_image);
    img_r_pub.publish(right_image);
    img_match_pub.publish(match_image);


    //camera point marker
    double x_world = frm[j].pose_c2w.translation().x();
    double y_world = frm[j].pose_c2w.translation().y();
    double z_world = frm[j].pose_c2w.translation().z();

    marker_cam_pose.position.x = x_world;
    marker_cam_pose.position.y = y_world;
    marker_cam_pose.position.z = z_world;



    marker_cam_tmp.pose = marker_cam_pose;
    marker_cam_tmp.id = static_cast<int>(j);

    marker_cam.markers.push_back(marker_cam_tmp);

    for(unsigned long u = 0; u < 500; u++){
      if( frm[j].coordinate_meter->x == 0.0 || frm[j].coordinate_meter->z > 200) continue;
      //if(j % 5 != 0) break;
      Eigen::Vector3d feature_point;
      feature_point.x() = frm[j].coordinate_meter->x;
      feature_point.y() = frm[j].coordinate_meter->y;
      feature_point.z() = frm[j].coordinate_meter->z;

      feature_point = frm[j].pose_c2w * feature_point; // feature on camera frame => feature on world frame
      double x_f_w = feature_point.x();
      double y_f_w = feature_point.y();
      double z_f_w = feature_point.z();

      marker_feature_pose.position.x = x_f_w;
      marker_feature_pose.position.y = y_f_w;
      marker_feature_pose.position.z = z_f_w;

      //draw feature point on map
      if(z_f_w > map.info.height || x_f_w-map.info.origin.position.x > map.info.width) continue;

      map.data[static_cast<unsigned long>(z_f_w)*static_cast<unsigned long>(map.info.width) +
          static_cast<unsigned long>(x_f_w-map.info.origin.position.x)] = 100;
      map.data[static_cast<unsigned long>(z_f_w)*static_cast<unsigned long>(map.info.width) +
          static_cast<unsigned long>(x_f_w-map.info.origin.position.x+1)] = 100;
      map.data[static_cast<unsigned long>(z_f_w)*static_cast<unsigned long>(map.info.width) +
          static_cast<unsigned long>(x_f_w-map.info.origin.position.x-1)] = 100;
      map.data[static_cast<unsigned long>(z_f_w+1)*static_cast<unsigned long>(map.info.width) +
          static_cast<unsigned long>(x_f_w-map.info.origin.position.x)] = 100;
      map.data[static_cast<unsigned long>(z_f_w-1)*static_cast<unsigned long>(map.info.width) +
          static_cast<unsigned long>(x_f_w-map.info.origin.position.x)] = 100;

      marker_feature_tmp.pose = marker_feature_pose;
      marker_feature_tmp.id = static_cast<int>(j*500 + u);

      //marker_feature.markers.push_back(marker_feature_tmp);
      //feature_count++;
      //if(feature_count > 50) break;

    }

    //draw camera point on map
    for(int i = -1; i < 2; i++){
      for(int k = -1; k < 2; k++){
        if (z_world+i < 0 || z_world+i > map.info.height || x_world-map.info.origin.position.x+k < 0
            || x_world-map.info.origin.position.x+k > map.info.width ) continue;

        map.data[static_cast<unsigned long>(z_world+i)*static_cast<unsigned long>(map.info.width) +
            static_cast<unsigned long>(x_world-map.info.origin.position.x+k)] = 0;
      }
    }

    if(j < frame_number-1)  j++;


    ros::spinOnce();
    camera_pose_pub.publish(marker_cam);
    //feature_pose_pub.publish(marker_feature);
    map_pub.publish(map);

    loop_rate.sleep();
  }



  delete[] frm;
  return 0;
}
