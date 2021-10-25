#include <filesystem>
#include <iostream>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>

struct Features_t
{
  inline Features_t(const cv::Mat& mask, double area) : area(area)
  {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    assert(contours.size() == 1);
    for (const auto& contour : contours)
    {
      std::vector<cv::Point> hull;
      cv::convexHull(contour, hull);
      cv::Point2f center;
      cv::minEnclosingCircle(contour, center, radius);
      auto min_area_rect = cv::minAreaRect(contour);
      min_area_rect_height = min_area_rect.size.height;
      min_area_rect_width = min_area_rect.size.width;
      min_area_rect_angle = min_area_rect.angle;
      min_area_rect_area = min_area_rect_width * min_area_rect_height;
      contour_area = cv::contourArea(contour);
      convex_hull_area = cv::contourArea(hull);
    }
  }
  inline bool isCrack() const
  {
    if (radius < 32)
    {
      return false;
    }
    if (contour_area > 0.7 * min_area_rect_area)
    {
      return false;
    }
    if (convex_hull_area > 0.9 * min_area_rect_area)
    {
      return false;
    }
    if (contour_area > 0.8 * convex_hull_area)
    {
      return false;
    }
    return true;
  }

  friend std::ostream& operator<<(std::ostream&, const Features_t&);

  float area;
  float radius;
  float contour_area;
  float convex_hull_area;
  float min_area_rect_area;
  float min_area_rect_angle;
  float min_area_rect_width;
  float min_area_rect_height;
};
inline std::ostream& operator<<(std::ostream& os, const Features_t& features)
{
  os << features.area << ','
     << features.radius << ','
     << features.contour_area << ','
     << features.convex_hull_area << ','
     << features.min_area_rect_area << ','
     << features.min_area_rect_angle << ','
     << features.min_area_rect_width << ','
     << features.min_area_rect_height << ',';
  return os;
}

class HSV
{
public:
  inline HSV(unsigned int H, double S = 1, double V = 1) : H(H), S(S), V(V) { assert(H < 360); }
  inline cv::Scalar toBGRA() const
  {
    return cv::Scalar(f(1) * 255, f(3) * 255, f(5) * 255, 255);
  }

private:
  double H;
  double S;
  double V;
  double f(int n) const
  {
    double k = (n + H / 60);
    while (k >= 6)
    {
      k -= 6;
    }
    return V - V * S * std::max(0.0, std::min(k, std::min(4.0 - k, 1.0)));
  }
};

class Model_t
{
private:
  cv::Mat img;
  cv::Mat orig;
  std::filesystem::path filename;

public:
  Model_t(const std::filesystem::path& input, const std::filesystem::path& output) : filename(output)
  {
    orig = cv::imread(input, cv::IMREAD_COLOR);
    assert(!orig.empty());

    cv::cvtColor(orig, img, cv::COLOR_BGR2GRAY);
    cv::cvtColor(orig, orig, cv::COLOR_BGR2BGRA);
  }
  ~Model_t()
  {
    if (img.channels() != 4)
    {
      cv::cvtColor(img, img, cv::COLOR_GRAY2BGRA);
    }
    assert(img.type() == orig.type());
    cv::hconcat(img, orig, img);
    cv::imwrite(filename, img);
  }

  std::vector<Features_t> Run()
  {
    img = img * 2;
    img = img | 0x0F;
    cv::adaptiveThreshold(img, img, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 17, 1);

    cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC4);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(img, labels, stats, centroids);
    std::vector<int> cracks;
    std::vector<Features_t> features;
    for (int label = 1; label < nLabels; label++)
    {
      auto left = stats.at<int32_t>(label, cv::CC_STAT_LEFT);
      auto top = stats.at<int32_t>(label, cv::CC_STAT_TOP);
      auto width = stats.at<int32_t>(label, cv::CC_STAT_WIDTH);
      auto height = stats.at<int32_t>(label, cv::CC_STAT_HEIGHT);
      auto area = stats.at<int32_t>(label, cv::CC_STAT_AREA);
      if (area <= 64)
      {
        continue;
      }
      if (height > 1240)
      {
        continue;
      }
      if (left + width < 64 || left > (520 - 64))
      {
        continue;
      }
      if (left < 64 && height > 1.2 * width)
      {
        continue;
      }
      if (left + width > (520 - 64) && height > 2 * width)
      {
        continue;
      }

      Features_t feature(labels == label, area);
      if (feature.isCrack())
      {
        cracks.push_back(label);
        features.push_back(feature);
      }
    }
    unsigned int color_gap = 360 / std::max(cracks.size(), 1ul);
    for (size_t i = 0; i < cracks.size(); i++)
    {
      output.setTo(HSV(i * color_gap).toBGRA(), labels == cracks[i]);
    }
    img = output;
    return features;
  }
};

int main()
{
  std::cout << CV_MAJOR_VERSION << std::endl;
  std::cout << CV_MINOR_VERSION << std::endl;
  size_t i = 0;
  std::vector<std::filesystem::path> files;
  std::filesystem::path output("/mnt/d/cv");
  for (const auto& iter : std::filesystem::directory_iterator(std::filesystem::absolute("../../../data/LCMS-Range-Shuffled/test/")))
  {
    files.push_back(iter.path());
  }
  std::vector<std::vector<Features_t>> results(files.size());
#pragma omp parallel for
  for (size_t i = 0; i < files.size(); i++)
  {
    const auto file = files[i];
    results[i] = 
      Model_t(file, output / file.filename().replace_extension("png")).Run();
    if (i % 100 == 0)
    {
      std::cerr << "Processed " << i << std::endl;
    }
  }
  for (size_t i = 0; i < files.size(); i++)
  {
    const auto file = files[i].filename();
    std::cout << (results[i].size() > 0) << std::endl;
  }
  return 0;
}
