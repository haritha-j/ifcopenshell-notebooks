#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ PointT;

int main (int argc, char** argv)
{
  // All the objects needed
  pcl::PCDReader reader;
  pcl::PassThrough<PointT> pass;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

  // Datasets
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients), coefficients_cylinder2 (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_cylinder2 (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

  std::string type = "pipe";

  // Output directory for cylinder point clouds and parameters
  std::string output_dir = "output/";
  boost::filesystem::create_directory(output_dir);

  // Output file for parameters
  std::ofstream param_file;
  param_file.open (type + "_params.txt");

  // Input directory for point clouds
  std::string input_dir = "input/";
  boost::filesystem::directory_iterator end_itr;

  for (boost::filesystem::directory_iterator itr(input_dir); itr != end_itr; ++itr)
  {
    if (boost::filesystem::is_regular_file(itr->path()))
    {
      // Read in the cloud data
      std::cout << itr->path().string() << std::endl;
      reader.read (itr->path().string(), *cloud);
      std::cerr << "PointCloud has: " << cloud->size () << " data points." << std::endl;
      param_file << itr->path().filename().string() << std::endl;
      // Estimate point normals
      ne.setSearchMethod (tree);
      ne.setInputCloud (cloud);
      ne.setKSearch (50);
      ne.compute (*cloud_normals);

      // Create the segmentation object for cylinder segmentation and set all the parameters
      seg.setOptimizeCoefficients (true);
      seg.setModelType (pcl::SACMODEL_CYLINDER);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setNormalDistanceWeight (0.1);
      seg.setMaxIterations (10000);
      seg.setDistanceThreshold (0.1);
      seg.setRadiusLimits (0.0, 5.0);

      seg.setInputCloud (cloud);
      seg.setInputNormals (cloud_normals);

      // Obtain the cylinder inliers and coefficients
      seg.segment (*inliers_cylinder, *coefficients_cylinder);

      // Write the cylinder inliers to disk
      extract.setInputCloud (cloud);
      extract.setIndices (inliers_cylinder);
      extract.setNegative (false);
      pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
      extract.filter (*cloud_cylinder);

      if (cloud_cylinder->points.empty ()) 
      {
        std::cerr << "Can't find the cylindrical component." << std::endl;
        param_file << std::endl;
        if (type == "tee")
          param_file << std::endl;
      }
      else
      {
        std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->size () << " data points." << std::endl;
        writer.write (output_dir + "primary_" + itr->path().filename().string(), *cloud_cylinder, false);
        for (auto const &c: coefficients_cylinder->values) {
            param_file << c << ",";
        }
        param_file << std::endl;

        // tee segmentation
        if (type == "tee")
        {
          // extract remaining points for second segmentation
          extract.setNegative (true);
          extract.filter (*cloud_filtered2);
          extract_normals.setNegative (true);
          extract_normals.setInputCloud (cloud_normals);
          extract_normals.setIndices (inliers_cylinder);
          extract_normals.filter (*cloud_normals2);

          // second segmentation
          seg.setInputCloud (cloud_filtered2);
          seg.setInputNormals (cloud_normals2);
          seg.segment (*inliers_cylinder2, *coefficients_cylinder2);

          // Write the cylinder inliers to disk
          extract.setInputCloud (cloud_filtered2);
          extract.setIndices (inliers_cylinder2);
          extract.setNegative (false);

          pcl::PointCloud<PointT>::Ptr cloud_cylinder2 (new pcl::PointCloud<PointT> ());
          extract.filter (*cloud_cylinder2);
          if (cloud_cylinder2->points.empty ()) 
          {
            std::cerr << "Can't find the second cylindrical component." << std::endl;
            param_file << std::endl;
          }
          else
          {
            std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder2->size () << " data points." << std::endl;
            writer.write (output_dir + "secondary_" + itr->path().filename().string(), *cloud_cylinder2, false);
            for (auto const &c: coefficients_cylinder2->values) {
              param_file << c << ",";
            }
            param_file << std::endl;
          }
        }
      }
    }
  }
  param_file.close();

  return (0);
}