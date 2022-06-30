#ifndef __OCR_UTILS_H__
#define __OCR_UTILS_H__

#include <opencv2/core.hpp>

#include <struct.h>

double getCurrentTime();

inline bool isFileExists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

std::wstring strToWstr(std::string str);

scaleParam getScaleParam(cv::Mat &src, const float scale);


#endif //__OCR_UTILS_H__