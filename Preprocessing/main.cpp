#include <filesystem>
#include <iostream>
#include <type_traits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>

class Model_t
{
public:
  Model_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : filename(filename.string() + "_" + tag + ".png")
  {
    cv::cvtColor(img, this->img, cv::COLOR_BGR2GRAY);
  }
  void Run()
  {
    run_model();
  }
  ~Model_t()
  {
    cv::imwrite(filename, img);
  }

protected:
  virtual void run_model() = 0;
  cv::Mat img;
  std::filesystem::path filename;
};

class Null_t : public Model_t
{
public:
  Null_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
  }
};

class Thresholding_t : public Model_t
{
public:
  Thresholding_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    img.convertTo(img, CV_32F);
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    cv::convertScaleAbs(img, img, 1.5, 0);
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  }
};

class Canny_t : public Model_t
{
public:
  Canny_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::Canny(img, img, 100, 100 * 2);
  }
};

class HistogramEqualization_t : public Model_t
{
public:
  HistogramEqualization_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::equalizeHist(img, img);
  }
};

class Sobel_t : public Model_t
{
public:
  Sobel_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    int ddepth = CV_32F;
    int ksize = 1;
    int scale = 1;
    int delta = 0;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(img, grad_x, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
    cv::Sobel(img, grad_y, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
    // converting back to CV_8U
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, img);
  }
};

class SIFT_t : public Model_t
{
public:
  SIFT_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::Ptr<cv::SIFT> pSIFT = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    pSIFT->detect(img, keypoints);
    cv::drawKeypoints(img, keypoints, img);
  }
};

class SURF_t : public Model_t
{
public:
  SURF_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::Ptr<cv::xfeatures2d::SURF> pSURF = cv::xfeatures2d::SURF::create();
    std::vector<cv::KeyPoint> keypoints;
    pSURF->detect(img, keypoints);
    cv::drawKeypoints(img, keypoints, img);
  }
};

class HIST_t : public Model_t
{
public:
  HIST_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    float range[] = {0, 256}; //the upper boundary is exclusive
    const float* histRange = {range};
    int histSize = 256;
    cv::Mat hist;
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, true);

    cv::Point maxLoc;
    cv::minMaxLoc(hist, NULL, NULL, NULL, &maxLoc);
    std::cout << maxLoc << std::endl;
    cv::normalize(hist, hist, 0, hist.rows, cv::NORM_MINMAX, -1, cv::Mat());

    int hist_w = 512, hist_h = 400;
    int bin_w = ((double)hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 1; i < histSize; i++)
    {
      cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - (hist.at<float>(i - 1))),
               cv::Point(bin_w * (i), hist_h - (hist.at<float>(i))),
               cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    img = histImage;
  }
};

class Kmeans_t : public Model_t
{
public:
  Kmeans_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    img *= 2;
    img &= 0xF0;
    img = ~img;
    int rows = img.rows;
    int cols = img.cols;
    img.convertTo(img, CV_32F);
    cv::Mat samples(img.rows * img.cols, 1, CV_32F);
    for (int y = 0; y < img.rows; y++)
      for (int x = 0; x < img.cols; x++)
        samples.at<float>(y + x * img.rows) = img.at<float>(y, x);
    cv::Mat labels, centers;
    cv::kmeans(samples, 4, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10, 1.0), 10, cv::KMEANS_PP_CENTERS, centers);

    cv::Mat cent = centers.reshape(1, centers.rows);
    // replace pixel values with their center value:
    for (size_t y = 0; y < img.rows; y++)
    {
      for (int x = 0; x < img.cols; x++)
      {
        int center_id = labels.at<int>(y + x * img.rows);
        img.at<float>(y, x) = 256 / centers.at<float>(center_id);
      }
    }
    img.convertTo(img, CV_8UC1);
    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    img = ~img;
  }
};

class AnisotropicDiffusion_t : public Model_t
{
public:
  AnisotropicDiffusion_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    img.convertTo(img, CV_32FC1, 1.0 / 255.0);
    for (int k = 0; k < 1000; k++)
    {
      for (int i = 1; i < img.rows - 1; i++)
      {
        for (int j = 1; j < img.cols - 1; j++)
        {
          float cN, cS, cE, cW;
          float deltacN, deltacS, deltacE, deltacW;

          deltacN = img.at<float>(i, j - 1) - img.at<float>(i, j);
          deltacS = img.at<float>(i, j + 1) - img.at<float>(i, j);
          deltacE = img.at<float>(i + 1, j) - img.at<float>(i, j);
          deltacW = img.at<float>(i - 1, j) - img.at<float>(i, j);

          cN = abs(exp(-1 * (deltacN * deltacN / (0.015 * 0.015))));
          cS = abs(exp(-1 * (deltacS * deltacS / (0.015 * 0.015))));
          cE = abs(exp(-1 * (deltacE * deltacE / (0.015 * 0.015))));
          cW = abs(exp(-1 * (deltacW * deltacW / (0.015 * 0.015))));

          img.at<float>(i, j) = img.at<float>(i, j) * (1 - 0.1 * (cN + cS + cE + cW)) +
                                0.1 * (cN * img.at<float>(i, j - 1) + cS * img.at<float>(i, j + 1) + cE * img.at<float>(i + 1, j) + cW * img.at<float>(i - 1, j));
        }
      }
    }
  }
};

class ConnectedComponent_t : public Model_t
{
public:
  ConnectedComponent_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats((img < 120) & (img > 0), labels, stats, centroids);
    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0); //background
    for (int label = 1; label < nLabels; label++)
    {
      int32_t area = stats.at<int32_t>(label, cv::CC_STAT_AREA);
      colors[label] = area < 100 ? cv::Vec3b(0, 0, 0) : /*cv::Vec3b(255, 255, 255); // */cv::Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
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
    img = dst;
  }
};

class TEST_t : public Model_t
{
public:
  TEST_t(const cv::Mat& img, const std::filesystem::path& filename, const std::string& tag) : Model_t(img, filename, tag) {}

protected:
  void run_model() override
  {
    cv::Mat tmp;
    img *= 2;
    img &= 0xF0;
    img ^= 0xFF;
  }
};

void ProcessImage(const std::filesystem::path& input, const std::filesystem::path& output)
{
  cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
  assert(!img.empty());

  ConnectedComponent_t(img, output, "ConnectedComponent").Run();
}

int main()
{
  size_t i = 0;
  std::vector<std::filesystem::path> files;
  std::filesystem::path output("/mnt/r/cv");
  for (const auto& iter : std::filesystem::directory_iterator("../../data/reindexed"))
  {
    files.push_back(iter.path());
    //if (i++ > 20)
    //  break;
  }
  #pragma omp parallel for
  for (const auto& file : files)
  {
    ProcessImage(file, output / file.stem());
  }
  return 0;
}
