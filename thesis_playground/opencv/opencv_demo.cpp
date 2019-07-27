#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(void)
{
　　Mat src_image_ = imread("test.jpg");   // 读取当前目录下的test.jpg图片
　　imshow("src_image_", src_image_);   // 打开图片文件
　　waitKey(5000);     // 停留5秒钟

　　return 0;
}
