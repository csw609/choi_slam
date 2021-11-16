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
  unsigned long frame_number = 50;

  ros::init(argc, argv, "test");
  ros::NodeHandle nh;

  sensor_msgs::Image left_image;
  sensor_msgs::Image right_image;
  sensor_msgs::Image match_image;
  cv_bridge::CvImage cv_bridge;

  //rviz visualization
  visualization_msgs::Marker marker_tmp;
  visualization_msgs::MarkerArray marker_a;
  visualization_msgs::MarkerArray marker_a_tmp;

  geometry_msgs::Pose marker_pose;
  marker_pose.orientation.w = 1.0;
  marker_pose.orientation.x = 0.0;
  marker_pose.orientation.y = 0.0;
  marker_pose.orientation.z = 0.0;
  marker_tmp.header.frame_id = "camera";
  marker_tmp.header.seq = 1;
  marker_tmp.header.stamp.sec = 0;
  marker_tmp.header.stamp.nsec = 0;
  marker_tmp.type = 2; //sphere
  marker_tmp.scale.x = 0.25;
  marker_tmp.scale.y = 0.25;
  marker_tmp.scale.z = 0.25;
  marker_tmp.color.r = 1.0;
  marker_tmp.color.g = 0.0;
  marker_tmp.color.b = 0.0;
  marker_tmp.color.a = 0.5;

  marker_tmp.id = 0;

  marker_a.markers.resize(frame_number);

  ros::Publisher img_l_pub = nh.advertise<sensor_msgs::Image>("left_image",1000);
  ros::Publisher img_r_pub = nh.advertise<sensor_msgs::Image>("right_image",1000);
  ros::Publisher img_match_pub = nh.advertise<sensor_msgs::Image>("match_image",1000);
  ros::Publisher camera_pose = nh.advertise<visualization_msgs::MarkerArray>("camera_pose",1000);

  //camera information//////////////////////////////
  double base_line_meter = 0.54;
  double fx = 7.215377 * 100;
  double fy = fx;
  double cx = 6.095593*100;
  double cy = 1.728540*100;
  double cam_pix_size = 4.65 * 0.000001;

  Eigen::Vector2d focal_length(fx,fy); // pixels
  Eigen::Vector2d principal_point(cx,cy);

  // ///////////////////////////////////////////////

  //optimization /////////////
  //pose optimization///////////////////////////////////
  // step 1. create linear solver
  std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver =
  g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();

  // step 2. create block solver
  std::unique_ptr<g2o::BlockSolver_6_3> block_solver =
  g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));

  // step 3. create optimization algorithm
  g2o::OptimizationAlgorithm* algorithm
  = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

  // step 4. create optimizer
  g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer;
  optimizer->setAlgorithm(algorithm);
  optimizer->setVerbose(false);	// to print optimization process

  // set up camera params
  g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],
                             principal_point[0],principal_point[1],
                             base_line_meter);
  // ///////////////////////////////////////////////////////////////


  //image input and processing
  //ROS_INFO("debug");

  choi::frame *frm;
  frm = new choi::frame[frame_number];

  for(unsigned long j = 0; j < frame_number; j++){
    //ROS_INFO("debug %d", j);
    std::string img_l_path;
    std::string img_r_path;
    if(j < 10){
      img_l_path = "/home/csw/cv/KITTI/039/image_00/data/000000000" + std::to_string(j) +".png";
      img_r_path = "/home/csw/cv/KITTI/039/image_01/data/000000000"+ std::to_string(j) +".png";
    }
    else if(j < 100){
      img_l_path = "/home/csw/cv/KITTI/039/image_00/data/00000000" + std::to_string(j) +".png";
      img_r_path = "/home/csw/cv/KITTI/039/image_01/data/00000000"+ std::to_string(j) +".png";
    }
    else if(j < 1000){
      img_l_path = "/home/csw/cv/KITTI/039/image_00/data/0000000" + std::to_string(j) +".png";
      img_r_path = "/home/csw/cv/KITTI/039/image_01/data/0000000"+ std::to_string(j) +".png";
    }

    cv::Mat img_left = cv::imread(img_l_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_right = cv::imread(img_r_path, CV_LOAD_IMAGE_GRAYSCALE);

    frm[j].img_l = img_left;
    frm[j].img_r = img_right;
    //ROS_INFO("debug1");
    frm[j].feature_extract();
    //ROS_INFO("debug2");
    frm[j].feature_match();
    //ROS_INFO("debug3");
    frm[j].sort_match(); // need
    //ROS_INFO("debug4");
    frm[j].triangulation();
    //ROS_INFO("debug5");
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

      cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING); //need improve BF this is not efficient
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
      Eigen::Vector3d prev2cur_trans;
      prev2cur_trans.x() = p_c[0]*cam_pix_size;
      prev2cur_trans.y() = p_c[1]*cam_pix_size;
      prev2cur_trans.z() = p_c[2]*cam_pix_size;
      Eigen::Matrix3d prev2cur_rot;

      cv::cv2eigen(R_c,prev2cur_rot);
      //ROS_INFO("x=%lf, y=%lf, z=%lf ,  frame%d", p_c[0]*cam_pix_size, p_c[1]*cam_pix_size, p_c[2]*cam_pix_size, static_cast<int>(j));

      //if reliable
      if(std::abs(prev2cur_trans.x()) < 2.0 && std::abs(prev2cur_trans.y()) < 0.5 && std::abs(prev2cur_trans.z()) < 2.0){
        frm[j].pose_prev = prev2cur_rot;
        frm[j].pose_prev.translation() = prev2cur_trans;
      }
      else{//constant velocity motion model
        frm[j].pose_prev = frm[j-1].pose_prev;
      }
      ROS_INFO("filtered x=%lf, y=%lf, z=%lf ,  frame%d", frm[j].pose_prev.translation().x(), frm[j].pose_prev.translation().y(),
               frm[j].pose_prev.translation().z(), static_cast<int>(j));

      frm[j].pose_c2w = frm[j-1].pose_c2w * frm[j].pose_prev;

      ROS_INFO("world x=%lf, y=%lf, z=%lf ,  frame%d", frm[j].pose_c2w.translation().x(), frm[j].pose_c2w.translation().y(),
               frm[j].pose_c2w.translation().z(), static_cast<int>(j));
    }

  }



  //ROS_INFO("debug");
  int test_num = 10;
  int point_id = test_num;
  for(int j = 0; j < test_num; j++){
    g2o::VertexSCam * v_se3
        = new g2o::VertexSCam();
    if(j == 0){
      v_se3->setId(j);
      v_se3->setEstimate(frm[j].pose_c2w);
      v_se3->setAll();
      v_se3->setFixed(true);
      optimizer->addVertex(v_se3);
    }
    else if(j > 0){
      v_se3->setId(j);
      v_se3->setEstimate(frm[j].pose_c2w);
      v_se3->setAll(); //-------------
      v_se3->setFixed(false);

      optimizer->addVertex(v_se3);
    }
  }

    int BA = 0;

    for(unsigned long j = 0; j < 10; j++){
      for(unsigned long i = 0; i < frm[j].matches_with_prev.size(); i++){
        if(BA >= 30){ //Number of points to use for BA
          break;
        }
        unsigned long idx_prev = static_cast<unsigned long>(frm[j].matches_with_prev[i].queryIdx);
        if(frm[j-1].coordinate_meter[idx_prev].x != 0.0 && frm[j-1].coordinate_meter[idx_prev].z < 30.0){

          g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();
          v_p->setId(point_id);
          v_p->setMarginalized(true);
          Eigen::Vector3d kp(static_cast<double>(frm[j-1].coordinate_meter[idx_prev].x/cam_pix_size)
                            ,static_cast<double>(frm[j-1].coordinate_meter[idx_prev].y/cam_pix_size)
                            ,static_cast<double>(frm[j-1].coordinate_meter[idx_prev].z/cam_pix_size));
          v_p->setEstimate(kp);

          int num_obs = 0;
          for(int k = 0; k < 10; k++){
            Eigen::Vector3d z;
            dynamic_cast<g2o::VertexSCam*>
              (optimizer->vertices().find(static_cast<int>(j))->second)
              ->mapPoint(z,kp); // vertex에 들어가있는 pose를 이용해서 point calculate stereo projection

            if (z[0]>=0 && z[1]>=0 && z[0]< IMAGE_WIDTH && z[1] < IMAGE_HEIGHT) //projection이 이미지 내에 된다면
            {
              ++num_obs;
            }
            if(num_obs >= 2) break;
          }

          if(num_obs >=2){
            optimizer->addVertex(v_p);

            for(int k = 0; k < 10; k++){
              Eigen::Vector3d z;

              dynamic_cast<g2o::VertexSCam*>
                (optimizer->vertices().find(static_cast<int>(j))->second)
                ->mapPoint(z,kp);

              if (z[0]>=0 && z[1]>=0 && z[0]< IMAGE_WIDTH && z[1] < IMAGE_HEIGHT){
                g2o::Edge_XYZ_VSC * e = new g2o::Edge_XYZ_VSC();
                e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);
                e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>      //edge에 pose vertex를 도착 vertex로
                              (optimizer->vertices().find(k)->second);
                e->setMeasurement(z);
                e->information() = Eigen::Matrix3d::Identity();
                optimizer->addEdge(e);
              }

            }
          }
          point_id++;
          BA++;
        }
      }
    }

  // 단위 맞추기 optimization 점검하기


  optimizer->initializeOptimization();

  ROS_INFO("saved?");
  optimizer->save("/home/csw/cv/before.g2o");
  ROS_INFO("saved?");

  optimizer->optimize(1000);
  Eigen::Vector3d vec[40];
  for(int i = 0; i < 10; i++){
    g2o::HyperGraph::VertexIDMap::iterator v_it
            = optimizer->vertices().find(i);

    //g2o::VertexPointXYZ * posed
    //        = dynamic_cast< g2o::VertexPointXYZ * > (v_it->second);
    g2o::VertexSCam * posed
            = dynamic_cast< g2o::VertexSCam * > (v_it->second);

    Eigen::Isometry3d second = posed->estimate();
    vec[i] = second.translation();
    ROS_INFO("%d : %lf  %lf   %lf", i, vec[i].x() * cam_pix_size, vec[i].y() * cam_pix_size, vec[i].z() * cam_pix_size);
    ROS_INFO("%d : %lf  %lf   %lf", i, vec[i].x(), vec[i].y(), vec[i].z());


  }

  ROS_INFO("saved?");
  optimizer->save("/home/csw/cv/after.g2o");
  ROS_INFO("saved?");

  //pose2_vec
  ros::Rate loop_rate(4);
  std_msgs::Header header;
  int count = 0;
  while (ros::ok())
  {

    std_msgs::String msg;
    msg.data = "hello world";
    frm[count].draw_feature_onframe();
    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm[count].img_l);
    cv_bridge.toImageMsg(left_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm[count].img_l);
    cv_bridge.toImageMsg(right_image);

    cv::Mat dst;
    frm[count].draw_match(dst);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, dst);
    cv_bridge.toImageMsg(match_image);

    left_image.header.stamp = ros::Time::now();
    left_image.header.frame_id = "camera";
    right_image.header.stamp = ros::Time::now();
    right_image.header.frame_id = "camera";
    match_image.header.stamp = ros::Time::now();
    match_image.header.frame_id = "camera";


    if(count < static_cast<int>(frame_number)-1){
      img_l_pub.publish(left_image);
      img_r_pub.publish(right_image);
      img_match_pub.publish(match_image);
      count++;
    }
    marker_a.markers.resize(count+1);

    for(int i = 0; i < count+1; i++){
      marker_pose.position.x = vec[count].x()* cam_pix_size * 0.2;
      marker_pose.position.y = vec[count].y()* cam_pix_size * 0.2;
      marker_pose.position.z = vec[count].z()* cam_pix_size * 0.2;

      marker_tmp.pose = marker_pose;
      marker_tmp.id = count;

      marker_a.markers[static_cast<unsigned long>(count)] = marker_tmp;
    }

    ros::spinOnce();
    camera_pose.publish(marker_a);

    loop_rate.sleep();
  }

  delete[] frm;
  return 0;
}
