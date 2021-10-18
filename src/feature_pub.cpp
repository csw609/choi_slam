#include "choi_cv.cpp"
int main(int argc, char **argv)
{
  ros::init(argc, argv, "test");
  ros::NodeHandle nh;

  sensor_msgs::Image left_image;
  sensor_msgs::Image right_image;
  sensor_msgs::Image match_image;
  cv_bridge::CvImage cv_bridge;

  cv::Mat img_l = cv::imread("/home/csw/cv/KITTI/left_image_00/data/0000000000.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_r = cv::imread("/home/csw/cv/KITTI/right_image_01/data/0000000000.png", CV_LOAD_IMAGE_GRAYSCALE);

  ros::Publisher img_l_pub = nh.advertise<sensor_msgs::Image>("left_image",1000);
  ros::Publisher img_r_pub = nh.advertise<sensor_msgs::Image>("right_image",1000);
  ros::Publisher img_match_pub = nh.advertise<sensor_msgs::Image>("match_image",1000);

  //cv::imshow("Display", img);

  //cv::Ptr<ｐ（
  ros::Rate loop_rate(10);

  std_msgs::Header header;

  choi::frame frm(img_l, img_r);
  cv::Mat dst;

  frm.feature_extract();
  frm.feature_match();
  frm.sort_match();
  frm.draw_match(dst);
  int left_idx  = frm.matches[400].queryIdx;
  int right_idx = frm.matches[400].trainIdx;
  ROS_INFO("left : %d, right : %d",left_idx, right_idx);
  float left_x  = frm.kp_l[left_idx].pt.x;
  float left_y  = frm.kp_l[left_idx].pt.y;
  float right_x = frm.kp_r[right_idx].pt.x;
  float right_y = frm.kp_r[right_idx].pt.y;
  ROS_INFO("l_x : %f, l_y : %f, r_x : %f, r_y : %f",left_x, left_y, right_x, right_y );

  frm.draw_byFeatureIdx_onframe(left_idx,right_idx);
  //frm.draw_feature_onframe();
  while (ros::ok())
  {
    std_msgs::String msg;
    msg.data = "hello world";
    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm.img_l);
    cv_bridge.toImageMsg(left_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm.img_r);
    cv_bridge.toImageMsg(right_image);

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

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
