#include "mio/mio.hpp"
#include "rapidxml-1.13/rapidxml.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <type_traits>
#include <vector>

void CCL(const cv::Mat& img, cv::Mat& mask)
{
  cv::Mat labels;
  cv::Mat stats;
  cv::Mat centroids;
  cv::Mat grey;
  cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
  int nLabels = cv::connectedComponentsWithStats((grey < 120) & (grey > 0), labels, stats, centroids);
  std::vector<cv::Vec3b> colors(nLabels);
  colors[0] = cv::Vec3b(0, 0, 0); //background
  for (int label = 1; label < nLabels; label++)
  {
    int32_t area = stats.at<int32_t>(label, cv::CC_STAT_AREA);
    colors[label] = area < 100 ? cv::Vec3b(0, 0, 0)
                               : cv::Vec3b(128 + (rand() & 127), 128 + (rand() & 127), 128 + (rand() & 127));
  }
  cv::Mat dst(img.size(), CV_8UC3);
  //cv::cvtColor(img, dst, cv::COLOR_GRAY2BGR);
  for (int r = 0; r < dst.rows; ++r)
  {
    for (int c = 0; c < dst.cols; ++c)
    {
      auto color = colors[labels.at<int>(r, c)];
      dst.at<cv::Vec3b>(r, c) = color;
    }
  }
  mask = dst;
}

const int HEIGHT = 1250;
const int WIDTH = 520;
const int UPPER = HEIGHT / 3;
const int LOWER = 2 * HEIGHT / 3;
class object_t
{
public:
  object_t(rapidxml::xml_node<>* node)
  {
    name = node->first_node("name")->value();
    rapidxml::xml_node<>* bndbox = node->first_node("bndbox");
    xmin = std::atoi(bndbox->first_node("xmin")->value());
    ymin = std::atoi(bndbox->first_node("ymin")->value());
    xmax = std::atoi(bndbox->first_node("xmax")->value());
    ymax = std::atoi(bndbox->first_node("ymax")->value());
  }
  bool is_upper() const
  {
    return ymax < UPPER;
  }
  bool is_middle() const
  {
    return (ymin >= UPPER) && (ymax < LOWER);
  }
  bool is_lower() const
  {
    return ymin >= LOWER;
  }
  bool has_upper() const
  {
    return ymin < UPPER;
  }
  bool has_middle() const
  {
    return (ymax >= UPPER) && (ymin < LOWER);
  }
  bool has_lower() const
  {
    return ymax >= LOWER;
  }
  object_t get_upper() const
  {
    return object_t(name,
                    xmin, ymin,
                    xmax, std::min(ymax, UPPER - 1));
  }
  object_t get_middle() const
  {
    return object_t(name,
                    xmin, std::max(ymin, UPPER),
                    xmax, std::min(ymax, LOWER - 1));
  }
  object_t get_lower() const
  {
    return object_t(name,
                    xmin, std::max(ymin, LOWER),
                    xmax, ymax);
  }
  operator cv::Rect() const
  {
    return cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  int xmin;
  int ymin;
  int xmax;
  int ymax;
  std::string name;

private:
  object_t(const std::string& name, int xmin, int ymin, int xmax, int ymax) : name(name),
                                                                              xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax)
  {
  }
};

class annotation_t
{
public:
  annotation_t(const std::filesystem::path& path)
  {
    mio::mmap_source mmap(path.string());
    rapidxml::xml_document<> doc;
    doc.parse<0>((char*)mmap.data());
    auto root = doc.first_node("annotation");
    filename = root->first_node("filename")->value();
    for (auto node = root->first_node("object"); node; node = node->next_sibling())
    {
      objects.emplace_back(node);
    }
  }
  std::string filename;
  std::vector<object_t> objects;
};

class image_t
{
public:
  image_t(const std::filesystem::path& path) : filename(path.stem())
  {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    assert(!img.empty());
    CCL(img, img);
    cv::cvtColor(img, mask, cv::COLOR_BGR2GRAY);
    this->img = img.clone();
  }
  bool has_img(const object_t& obj)
  {
    return cv::countNonZero(mask(obj)) != 0;
  }
  void draw(const object_t& obj)
  {
    cv::rectangle(img,
                  cv::Point(obj.xmin, obj.ymin),
                  cv::Point(obj.xmax, obj.ymax),
                  cv::Scalar(0, 0, 255));
  }
  void save(const std::filesystem::path& path)
  {
    cv::line(img,
             cv::Point(0, UPPER),
             cv::Point(520, UPPER),
             cv::Scalar(255, 0, 255));
    cv::line(img,
             cv::Point(0, LOWER),
             cv::Point(520, LOWER),
             cv::Scalar(255, 0, 255));
    cv::imwrite((path / filename).replace_extension("png"), img);
  }
  cv::Mat img;
  cv::Mat mask;
  std::filesystem::path filename;
};

class label_t
{
public:
  label_t() : filename(),
              upper(false), middle(false), lower(false)
  {
  }
  bool upper;
  bool middle;
  bool lower;
  std::string filename;
};
std::ostream& operator<<(std::ostream& os, const label_t& label)
{
  os << label.filename << ',' << label.upper << ',' << label.middle << ',' << label.lower;
  return os;
}

int main()
{
  size_t i = 0;
  std::vector<std::filesystem::path> files;
  std::filesystem::path output("/mnt/r/cv");
  std::filesystem::path input("../../../data/training");
  for (const auto& iter : std::filesystem::directory_iterator(input / "annotations"))
  {
    files.push_back(iter.path());
  }
  std::vector<label_t> labels(files.size());

#pragma omp parallel for
  for (size_t i = 0; i < files.size(); i++)
  {
    const auto& file = files[i];
    auto& label = labels[i];
    annotation_t annotation(file);
    label.filename = annotation.filename;
    image_t img(input / "images" / annotation.filename);
    for (const auto& obj : annotation.objects)
    {
      if (obj.has_upper())
      {
        auto tmp = obj.get_upper();
        if (img.has_img(tmp))
        {
          label.upper = true;
          img.draw(tmp);
        }
      }
      if (obj.has_middle())
      {
        auto tmp = obj.get_middle();
        if (img.has_img(tmp))
        {
          label.middle = true;
          img.draw(tmp);
        }
      }
      if (obj.has_lower())
      {
        auto tmp = obj.get_lower();
        if (img.has_img(tmp))
        {
          label.lower = true;
          img.draw(tmp);
        }
      }
    }
    img.save(output);
  }
  std::cout << "filename,upper,middle,lower" << std::endl;
  for (const auto& label : labels)
  {
    std::cout << label << std::endl;
  }
  return 0;
}
