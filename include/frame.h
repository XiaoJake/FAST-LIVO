// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_FRAME_H_
#define SVO_FRAME_H_

#include <common_lib.h>
#include <vikit/math_utils.h>
#include <vikit/abstract_camera.h>
#include <boost/noncopyable.hpp>

namespace lidar_selection {

struct Feature;
typedef std::shared_ptr<Feature>  FeaturePtr;

class Point;
typedef std::shared_ptr<Point> PointPtr;

typedef list<FeaturePtr> Features;
typedef vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
  static int                    frame_counter_;         //!< Counts the number of created frames. Used to set the unique id.
  int                           id_;                    //!< Unique id of the frame.
  // double                        timestamp_;             //!< Timestamp of when the image was recorded.
  vk::AbstractCamera*           cam_;                   //!< Camera model.
  SE3                           T_f_w_;                 //!< Transform (f)rame from (w)orld.
  Matrix<double, 6, 6>          Cov_;                   //!< Covariance.
  ImgPyr                        img_pyr_;               //!< Image Pyramid.
  Features                      fts_;                   //!< List of features in the image.
  vector<FeaturePtr>              key_pts_;               //!< Five features and associated 3D points which are used to detect if two frames have overlapping field of view.
  bool                          is_keyframe_;           //!< Was this frames selected as keyframe

  static void resetStatics();
  Frame(vk::AbstractCamera* cam, const cv::Mat& img);
  ~Frame();

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat& img);

  /// Select this frame as keyframe.
  void setKeyframe();

  /// Add a feature to the image
  void addFeature(FeaturePtr ftr);

  /// The KeyPoints are those five features which are closest to the 4 image corners
  /// and to the center and which have a 3D point assigned. These points are used
  /// to quickly check whether two frames have overlapping field of view.
  void setKeyPoints();

  /// Check if we can select five better key-points.
  void checkKeyPoints(FeaturePtr ftr);

  /// If a point is deleted, we must remove the corresponding key-point.
  void removeKeyPoint(FeaturePtr ftr);

  /// Return number of point observations.
  inline size_t nObs() const { return fts_.size(); }

  /// Check if a point in (w)orld coordinate frame is visible in the image.
  bool isVisible(const Vector3d& xyz_w) const;

  /// Full resolution image stored in the frame.
  inline const cv::Mat& img() const { return img_pyr_[0]; }

  /// Was this frame selected as keyframe?
  inline bool isKeyframe() const { return is_keyframe_; }

  /// Transforms point coordinates in world-frame (w) to camera pixel coordinates (c).
  inline Vector2d w2c(const Vector3d& xyz_w) const { return cam_->world2cam( T_f_w_ * xyz_w ); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const Vector2d& px) const { return cam_->cam2world(px[0], px[1]); }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const double x, const double y) const { return cam_->cam2world(x, y); }

  /// Transforms point coordinates in world-frame (w) to camera-frams (f).
  inline Vector3d w2f(const Vector3d& xyz_w) const { return T_f_w_ * xyz_w; }

  /// Transforms point from frame unit sphere (f) frame to world coordinate frame (w).
  inline Vector3d f2w(const Vector3d& f) const { return T_f_w_.inverse() * f; }

  /// Projects Point from unit sphere (f) in camera pixels (c).
  inline Vector2d f2c(const Vector3d& f) const { return cam_->world2cam( f ); }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }

  /// Frame jacobian for projection of 3D point in (f)rame coordinate to
  /// unit plane coordinates uv (focal length = 1).
  inline static void jacobian_xyz2uv_change(
      const Vector3d& xyz_in_world,
      const Vector3d& xyz_in_f,
      Matrix<double,2,6>& J,
      SE3& Tbc,
      SE3& T_ref_w,
      double fx)
  {
    //Vector3d xyz_in_imu = Tbc * xyz_in_world;
    //Vector3d xyz_in_imu = xyz_in_world;
    const double x = xyz_in_f[0];
    const double y = xyz_in_f[1];
    //const double z = xyz_in_f[2]; 
    //! minus symbol is added here.
    const double z_inv = 1./xyz_in_f[2];
    const double z_inv_2 = z_inv*z_inv;

    const double x_in_world = xyz_in_world[0];
    const double y_in_world = xyz_in_world[1];
    const double z_in_world = xyz_in_world[2];

    Matrix<double,2,3> J1;
    Matrix<double,3,6> J2;

    J1(0,0) = -fx * z_inv;              
    J1(0,1) = 0.0;              
    J1(0,2) = fx * x * z_inv_2;           

    J1(1,0) = 0.0;           
    J1(1,1) = -fx * z_inv;           
    J1(1,2) = fx * y * z_inv_2;         

    J2(0,0) = 1.0;             
    J2(0,1) = 0.0;                 
    J2(0,2) = 0.0;           
    J2(0,3) = 0.0;            
    J2(0,4) = z_in_world;   
    J2(0,5) = -y_in_world;        

    J2(1,0) = 0.0;               
    J2(1,1) = 1.0;            
    J2(1,2) = 0.0;          
    J2(1,3) = -z_in_world;     
    J2(1,4) = 0.0;             
    J2(1,5) = x_in_world;          

    J2(2,0) = 0.0;      
    J2(2,1) = 0.0;       
    J2(2,2) = 1.0;        
    J2(2,3) = y_in_world;  
    J2(2,4) = -x_in_world;       
    J2(2,5) = 0.0;   
    
    J = J1 * T_ref_w.rotation_matrix() * J2;// * J2;   
  }

  inline static void jacobian_xyz2uv(
    const Vector3d& xyz_in_f,
    Matrix<double,2,6>& J)
  {
    const double x = xyz_in_f[0];
    const double y = xyz_in_f[1];
    const double z_inv = 1./xyz_in_f[2];
    const double z_inv_2 = z_inv*z_inv;

    J(0,0) = -z_inv;              // -1/z
    J(0,1) = 0.0;                 // 0
    J(0,2) = x*z_inv_2;           // x/z^2
    J(0,3) = y*J(0,2);            // x*y/z^2
    J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
    J(0,5) = y*z_inv;             // y/z

    J(1,0) = 0.0;                 // 0
    J(1,1) = -z_inv;              // -1/z
    J(1,2) = y*z_inv_2;           // y/z^2
    J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
    J(1,4) = -J(0,3);             // -x*y/z^2
    J(1,5) = -x*z_inv;            // x/z
  }
};

typedef std::shared_ptr<Frame> FramePtr;

/// Some helper functions for the frame object.
namespace frame_utils {

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

/// Get the average depth of the features in the image.
bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min);

} // namespace frame_utils
} // namespace svo

#endif // SVO_FRAME_H_
