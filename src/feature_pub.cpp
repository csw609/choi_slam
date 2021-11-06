#include "vision/choi_cv.cpp"



// #include "g2o/types/sba/types_six_dof_expmap.h"

// G2O_USE_TYPE_GROUP(slam2d);
// G2O_USE_TYPE_GROUP(slam3d);

int main(int argc, char **argv)
{
  ros::init(argc, argv, "test");
  ros::NodeHandle nh;

  sensor_msgs::Image left_image;
  sensor_msgs::Image right_image;
  sensor_msgs::Image match_image;
  cv_bridge::CvImage cv_bridge;

  ros::Rate loop_rate(10);

  ros::Publisher img_l_pub = nh.advertise<sensor_msgs::Image>("left_image",1000);
  ros::Publisher img_r_pub = nh.advertise<sensor_msgs::Image>("right_image",1000);
  ros::Publisher img_match_pub = nh.advertise<sensor_msgs::Image>("match_image",1000);

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
  optimizer->setVerbose(true);	// to print optimization process

  // set up camera params
  g2o::VertexSCam::setKcam(focal_length[0],focal_length[1],
                             principal_point[0],principal_point[1],
                             base_line_meter);
  // ///////////////////////////////////////////////////////////////


  //image input and processing
  ROS_INFO("debug");
  choi::frame frm[20];
  for(int j = 0; j < 10; j++){
    ROS_INFO("debug %d", j);
    std::string img_l_path = "/home/csw/cv/KITTI/left_image_00/data/000000000" + std::to_string(j) +".png";
    std::string img_r_path = "/home/csw/cv/KITTI/right_image_01/data/000000000"+ std::to_string(j) +".png";

    cv::Mat img_left = cv::imread(img_l_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_right = cv::imread(img_r_path, CV_LOAD_IMAGE_GRAYSCALE);

    frm[j].img_l = img_left;
    frm[j].img_r = img_right;
    ROS_INFO("debug1");
    frm[j].feature_extract();
    ROS_INFO("debug2");
    frm[j].feature_match();
    ROS_INFO("debug3");
    frm[j].sort_match(); // need?
    ROS_INFO("debug4");
    frm[j].triangulation();
    ROS_INFO("debug5");
    if(j == 0){
      frm[0].trans_cam = Eigen::Vector3d(0,0,0);
      frm[0].q_cam     = Eigen::Quaterniond::Identity();
      frm[0].init_pose();
    }

  }


  ROS_INFO("debug");
  int point_id = 10;
  for(int j = 0; j < 10; j++){
    g2o::VertexSCam * v_se3
        = new g2o::VertexSCam();
    if(j == 0){
      v_se3->setId(j);
      v_se3->setEstimate(frm[j].pose_cam);
      v_se3->setAll();
      v_se3->setFixed(true);
      optimizer->addVertex(v_se3);
    }
    else if(j > 0){
      Eigen::Vector3d trans(0,0,0);
      Eigen::Quaterniond q;
      q.setIdentity();
      Eigen::Isometry3d pose;
      pose = q;
      pose.translation() = trans;
      v_se3->setId(j);
      v_se3->setEstimate(pose);
      v_se3->setAll(); //-------------
      v_se3->setFixed(false);

      optimizer->addVertex(v_se3);

      std::vector<cv::DMatch> matches_frame; //matching container
      cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING); //need improve BF not efficient
      matcher->match(frm[j-1].des_l,frm[j].des_l,matches_frame);

      std::sort(matches_frame.begin(),matches_frame.end());
      std::vector<cv::DMatch> good(matches_frame.begin(), matches_frame.begin() + 400);

      int BA = 0;
      for(unsigned long i = 0; i < good.size(); i++){
        if(BA >= 50){ //Number of points to use for BA
          break;
        }
        if(frm[j-1].coordinate_meter[static_cast<unsigned long>(good[i].queryIdx)].x != 0.0){
          g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();
          v_p->setId(point_id);
          v_p->setMarginalized(true);
          Eigen::Vector3d kp(static_cast<double>(frm[j-1].coordinate_meter[static_cast<unsigned long>(good[i].queryIdx)].x/cam_pix_size)
                            ,static_cast<double>(frm[j-1].coordinate_meter[static_cast<unsigned long>(good[i].queryIdx)].y/cam_pix_size)
                            ,static_cast<double>(frm[j-1].coordinate_meter[static_cast<unsigned long>(good[i].queryIdx)].z/cam_pix_size));
          v_p->setEstimate(kp);
          v_p->setFixed(true);

          optimizer->addVertex(v_p);
          //edge first pose, point
          Eigen::Vector3d z(static_cast<double>(frm[j-1].kp_l[static_cast<unsigned long>(good[i].queryIdx)].pt.x),
                            static_cast<double>(frm[j-1].kp_l[static_cast<unsigned long>(good[i].queryIdx)].pt.y),
                            0.0);

          g2o::Edge_XYZ_VSC * e //point에서 카메라 포즈까지의 edge
            = new g2o::Edge_XYZ_VSC();

          e->vertices()[0]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p); //edge에 point vertex를 출발 vertex로
          e->vertices()[1]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>      //edge에 pose vertex를 도착 vertex로
            (optimizer->vertices().find(j-1)->second);
          e->setMeasurement(z);
          //e->setParameterId(0,0);
          Eigen::Matrix3d info_matrix = Eigen::Matrix3d::Identity(3,3);
          e->setInformation(info_matrix);

          optimizer->addEdge(e); //엣지 추가

          //edge second pose, point

          Eigen::Vector3d z2(static_cast<double>(frm[j].kp_l[static_cast<unsigned long>(good[i].trainIdx)].pt.x),
                            static_cast<double>(frm[j].kp_l[static_cast<unsigned long>(good[i].trainIdx)].pt.y),
                            0.0);

          //ROS_INFO("x : %lf", static_cast<double>(frm2.kp_l[static_cast<unsigned long>(good[i].trainIdx)].pt.x));

          g2o::Edge_XYZ_VSC * e2 //point에서 카메라 포즈까지의 edge
            = new g2o::Edge_XYZ_VSC();

          e2->vertices()[0]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p); //edge에 point vertex를 출발 vertex로
          e2->vertices()[1]
            = dynamic_cast<g2o::OptimizableGraph::Vertex*>      //edge에 pose vertex를 도착 vertex로
            (optimizer->vertices().find(j)->second);
          e2->setMeasurement(z2);
          //e2->setParameterId(0,0);
          Eigen::Matrix3d info_matrix2 = Eigen::Matrix3d::Identity(3,3);
          e2->setInformation(info_matrix2);

          optimizer->addEdge(e2); //엣지 추가
          point_id++;
          BA++;
        }
      }
    }
  }


  optimizer->initializeOptimization();

  ROS_INFO("saved?");
  optimizer->save("/home/csw/cv/before.g2o");
  ROS_INFO("saved?");

  optimizer->optimize(1000);

  for(int i = 0; i < 10; i++){
    g2o::HyperGraph::VertexIDMap::iterator v_it
            = optimizer->vertices().find(j);

    //g2o::VertexPointXYZ * posed
    //        = dynamic_cast< g2o::VertexPointXYZ * > (v_it->second);
    g2o::VertexSCam * posed
            = dynamic_cast< g2o::VertexSCam * > (v_it->second);

    Eigen::Isometry3d second = posed->estimate();
    Eigen::Vector3d vec = second.translation();
    ROS_INFO("%lf  %lf   %lf",vec.x() * cam_pix_size, vec.y() * cam_pix_size, vec.z() * cam_pix_size);
    ROS_INFO("%lf  %lf   %lf",vec.x(), vec.y(), vec.z());
  }

  ROS_INFO("saved?");
  optimizer->save("/home/csw/cv/after.g2o");
  ROS_INFO("saved?");

  //pose2_vec

  std_msgs::Header header;
  while (ros::ok())
  {
    std_msgs::String msg;
    msg.data = "hello world";
    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm[0].img_l);
    cv_bridge.toImageMsg(left_image);

    cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::MONO8, frm[9].img_l);
    cv_bridge.toImageMsg(right_image);

    //cv_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::BGR8, dst);
    //cv_bridge.toImageMsg(match_image);

    left_image.header.stamp = ros::Time::now();
    left_image.header.frame_id = "camera";
    right_image.header.stamp = ros::Time::now();
    right_image.header.frame_id = "camera";
    match_image.header.stamp = ros::Time::now();
    match_image.header.frame_id = "camera";

    img_l_pub.publish(left_image);
    img_r_pub.publish(right_image);
    //img_match_pub.publish(match_image);

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
