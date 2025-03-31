# RS-FAST-LIVO

[English Version](README.md)

## 1. 介绍

本工程将最先进的LiDAR-惯性-视觉里程计系统`FAST-LIVO`适配了速腾聚创激光的新产品Active Camera.

**FAST-LIVO** 是一个快速的LiDAR惯性视觉里程计系统，它建立在两个紧密耦合和直接的里程计子系统之上：VIO子系统和LIO子系统。LIO子系统将新一帧点云（而不是边缘或平面特征点）配准到增量构建的点云地图中。地图点还关联了图像patch，VIO子系统通过最小化patch的直接光度误差来匹配新图像，而无需提取任何视觉特征（例如ORB或FAST角特征）。

相比原始工程, 做出的优化改进如下:
- 增加了初始化模块
- 在选点部分更严谨的策略（比如邻域光度协方差和响应值）
- 视觉地图点根据距离/夹角的精准noise建模
- 非极大值抑制选择特征
- LiDAR退化检测
- 点云去噪去拉丝点去拖影点

<div align="center">
    <img src="img/Framework.svg" width = 100% >
</div>

如果您需要有关`FAST-LIVO`算法的更多信息，可以参考官方仓库和维护者：
- **仓库**: <https://github.com/hku-mars/FAST-LIVO>
- **维护者**: [Chunran Zheng 郑纯然](https://github.com/xuankuzcr)， [Qingyan Zhu 朱清岩](https://github.com/ZQYKAWAYI)， [Wei Xu 徐威](https://github.com/XW-HKU)

## 2. 样例

### 2.1 使用 Active Camera 数据

<div align="center">
    <img src="img/hitsz.png" alt="mesh" />
    <p style="margin-top: 2px;">"HIT SZ Wall" 数据集. 左图: 原始图像, 右图: 建图结果</p>
</div>

<div align="center">
    <img src="img/hitsz_full.png" alt="mesh" />
    <p style="margin-top: 2px;">"HIT SZ Wall" 数据集. 完整建图结果</p>
</div>

<div align="center">
    <img src="img/robosense_logo.png" alt="mesh" />
    <p style="margin-top: 2px;"> "indoor robosense logo" 数据集. 左图: 原始图像, 右图: 完整建图结果 </p>
</div>

## 3. 依赖

### 3.1 ROS2

此项目基于 `ros2 humble`进行开发测试

根据您的操作系统选择 [官方教程](https://fishros.org/doc/ros2/humble/Installation.html) 中的指定内容进行执行

### 3.2 Sophus

Sophus 需要安装 non-templated/double-only 版本.

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus
git checkout a621ff
mkdir build && cd build && cmake ..
make
sudo make install
```

## 4. 安装编译

下载仓库：

将本工程代码拉取到一个`ros2`工作空间中，然后在终端中执行以下命令编译安装：
您可以创建一个新的文件夹或进入您现有的 `ros2` 工作空间，执行以下命令将代码拉取到工作空间内

```bash
colcon build --symlink-install
```

编译安装完成后，推荐刷新一下工作空间的 `bash profile`，确保组件功能正常

```bash
source install/setup.bash
```

## 4. 运行

### 4.1 重要参数

编辑 `config/RS_META.yaml` 来设置一些重要参数：

#### 4.1.1 算法

- `lid_topic`: LiDAR 话题
- `imu_topic`: IMU 话题
- `img_topic`: camera 话题
- `img_enable`: 开启 vio 子系统
- `lidar_enable`: 开启 lio 子系统
- `outlier_threshold`: 单个像素的光度误差（平方）的异常阈值。建议暗场景使用“50~250”，亮场景使用“500~1000”。该值越小，vio子模块的速度越快，但抗退化力越弱。
- `img_point_cov`: 每像素光度误差的协方差。
- `laser_point_cov`: 每个点的点对平面重新方差。
- `filter_size_surf`：对新扫描中的点进行降采样。建议室内场景为`0.05~0.15`，室外场景为`0.3~0.5`。
- `filter_size_map`：对LiDAR全局地图中的点进行降采样。建议室内场景为`0.15~0.3`，室外场景为`0.4~0.5`。
- `pcd_save_en`：如果为`true`，则将点云保存到pcd文件夹。如果`img_enable`为`1`，则保存RGB彩色点；如果`img_enable`是`0`，则按点云强度存储彩色点。
- `delta_time`：相机和激光雷达之间的时间偏移，用于校正时间戳错位。

在此之后，您可以直接在数据集上运行**RS-FAST-LIVO**。

#### 4.1.2 外参、内参

- `extrinsic_T`: 将IMU坐标系变换到LiDAR坐标系的平移
- `extrinsic_R:`: 将IMU坐标系变换到LiDAR坐标系的旋转
- `Rcl`: 将camera坐标系变换到LiDAR坐标系的平移
- `Pcl`: 将camera坐标系变换到LiDAR坐标系的旋转
- `camera_pinhole_rs.yaml`: camera内参

### 4.2 在数据集运行

从OneDrive ([FAST-LIVO-Datasets](TODO))下载数据集， 共包含 **xx** 个rosbag

```bash
ros2 run slam slam_node
```

## 5. 致谢

感谢 [FAST-LIVO](https://github.com/hku-mars/FAST-LIVO)、 [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) 和 [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)

## 6. License

该仓库在 [**GPLv2**](http://www.gnu.org/licenses/) 协议下开源