
#include <iostream>
#include "paddle_api.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "paddle_place.h"
#include "paddle_place.h"
#include "cls_process.h"
#include "crnn_process.h"
#include "db_post_process.h"
#include "arm_neon.h"

using namespace paddle::lite_api;

// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
//void NeonMeanScale(const float *din, float *dout, int size,
//                   const std::vector<float> mean,
//                   const std::vector<float> scale) {
//    if (mean.size() != 3 || scale.size() != 3) {
//        std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
//        exit(1);
//    }
//    float32x4_t vmean0 = vdupq_n_f32(mean[0]);
//    float32x4_t vmean1 = vdupq_n_f32(mean[1]);
//    float32x4_t vmean2 = vdupq_n_f32(mean[2]);
//    float32x4_t vscale0 = vdupq_n_f32(scale[0]);
//    float32x4_t vscale1 = vdupq_n_f32(scale[1]);
//    float32x4_t vscale2 = vdupq_n_f32(scale[2]);
//
//    float *dout_c0 = dout;
//    float *dout_c1 = dout + size;
//    float *dout_c2 = dout + size * 2;
//
//    int i = 0;
//    for (; i < size - 3; i += 4) {
//        float32x4x3_t vin3 = vld3q_f32(din);
//        float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
//        float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
//        float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
//        float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
//        float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
//        float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
//        vst1q_f32(dout_c0, vs0);
//        vst1q_f32(dout_c1, vs1);
//        vst1q_f32(dout_c2, vs2);
//
//        din += 12;
//        dout_c0 += 4;
//        dout_c1 += 4;
//        dout_c2 += 4;
//    }
//    for (; i < size; i++) {
//        *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
//        *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
//        *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
//    }
//}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes) {
    cv::Point rook_points[boxes.size()][4];
    for (int n = 0; n < boxes.size(); n++) {
        for (int m = 0; m < boxes[0].size(); m++) {
            rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                          static_cast<int>(boxes[n][m][1]));
        }
    }
    cv::Mat img_vis;
    srcimg.copyTo(img_vis);
    for (int n = 0; n < boxes.size(); n++) {
        const cv::Point *ppt[1] = {rook_points[n]};
        int npt[] = {4};
        cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
    }

    cv::imwrite("/Users/yang/CLionProjects/test_paddle_lite/data/images/vis.jpg", img_vis);
    std::cout << "The detection visualized image saved in ./vis.jpg" << std::endl;
    return img_vis;
}


void MeanScale(const float *din, float *dout, int size,
               const std::vector<float> mean,
               const std::vector<float> scale) {
    if (mean.size() != 3 || scale.size() != 3) {
        std::cerr << "[ERROR] mean or scale size must equal to 3" << std::endl;
        exit(1);
    }

    for (int i = 0; i < size; ++i) {
        dout[i] = (din[i] - mean[0]) * scale[0];
    }

    for (int i = size; i < 2 * size; ++i) {
        dout[i] = (din[i] - mean[1]) * scale[1];
    }

    for (int i = 2 * size; i < 3 * size; ++i) {
        dout[i] = (din[i] - mean[2]) * scale[2];
    }
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
    std::vector<std::string> res;
    if ("" == str)
        return res;
    char *strs = new char[str.length() + 1];
    std::strcpy(strs, str.c_str());

    char *d = new char[delim.length() + 1];
    std::strcpy(d, delim.c_str());

    char *p = std::strtok(strs, d);
    while (p) {
        std::string s = p;
        res.push_back(s);
        p = std::strtok(NULL, d);
    }

    return res;
}

std::map<std::string, double> LoadConfigTxt(std::string& config_path) {
    auto config = ReadDict(config_path);

    std::map<std::string, double> dict;
    for (int i = 0; i < config.size(); i++) {
        std::vector<std::string> res = split(config[i], " ");
        dict[res[0]] = stod(res[1]);
    }
    return dict;
}

cv::Mat DetResizeImg(const cv::Mat img, int max_size_len,
                     std::vector<float> &ratio_hw) {
    int w = img.cols;
    int h = img.rows;

    float ratio = 1.f;
    int max_wh = w >= h ? w : h;
    if (max_wh > max_size_len) {
        if (h > w) {
            ratio = static_cast<float>(max_size_len) / static_cast<float>(h);
        } else {
            ratio = static_cast<float>(max_size_len) / static_cast<float>(w);
        }
    }

    int resize_h = static_cast<int>(float(h) * ratio);
    int resize_w = static_cast<int>(float(w) * ratio);
    if (resize_h % 32 == 0)
        resize_h = resize_h;
    else if (resize_h / 32 < 1 + 1e-5)
        resize_h = 32;
    else
        resize_h = (resize_h / 32 - 1) * 32;

    if (resize_w % 32 == 0)
        resize_w = resize_w;
    else if (resize_w / 32 < 1 + 1e-5)
        resize_w = 32;
    else
        resize_w = (resize_w / 32 - 1) * 32;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));

    ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
    ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
    return resize_img;
}

int test_paddle_ocr_v4_paddle_lite() {
    std::string det_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_PP-OCRv4_det_opt.nb");
    std::string rec_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_PP-OCRv4_rec_opt.nb");
    std::string cls_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_ppocr_mobile_v2.0_cls_opt.nb");
    std::string ppocr_keys_v1_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/paddle_ocr_lib/ppocr_keys_v1.txt");
    std::string config_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/paddle_ocr_lib/config.txt");
    std::string image_file("/Users/yang/CLionProjects/test_paddle_lite/data/images/insurance.png");

    auto Config = LoadConfigTxt(config_file);
    int max_side_len = int(Config["max_side_len"]);
    int det_db_use_dilate = int(Config["det_db_use_dilate"]);

    cv::Mat img = cv::imread(image_file, cv::IMREAD_COLOR);
    std::vector<float> ratio_hw;
    cv::Mat srcimg;
    img.copyTo(srcimg);
    img = DetResizeImg(img, max_side_len, ratio_hw);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);
//    cv::resize(image, image_resize, cv::Size(640, 640));
//    image_resize.convertTo(image_resize, CV_32F, 1.0/255.0);
//    float w_scale = (float)image.cols / (float)640;
//    float h_scale = (float)image.rows / (float)640;

//    cv::transposeND(image, {2,0,1}, image_t);
//    cv::imshow("", image);
//    cv::waitKey(0);


    std::cout << "run paddleocr." << std::endl;
    // 1. Set MobileConfig
    MobileConfig config;
    // 2. Set the path to the model generated by opt tools
    config.set_model_from_file(det_model_file);
    // 3. Create PaddlePredictor by MobileConfig
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

    // input
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor0->mutable_data<float>();
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
//    NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
    MeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);

    // infer
    predictor->Run();

    // output
    std::unique_ptr<const Tensor> output_tensor0(
            std::move(predictor->GetOutput(0)));
    auto *predict_batch = output_tensor0->data<float>();
    auto predict_shape = output_tensor0->shape();
    std::cout << "output shape: [" << predict_shape[0] << ","
                                   << predict_shape[1] << ","
                                   << predict_shape[2] << ","
                                   << predict_shape[3] << "]" << std::endl;

    // decode
    // Get output and post process
    std::unique_ptr<const Tensor> output_tensor(
            std::move(predictor->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();

    // Save output
    float pred[shape_out[2] * shape_out[3]];
    unsigned char cbuf[shape_out[2] * shape_out[3]];

    for (int i = 0; i < int(shape_out[2] * shape_out[3]); i++) {
        pred[i] = static_cast<float>(outptr[i]);
        cbuf[i] = static_cast<unsigned char>((outptr[i]) * 255);
    }

    cv::Mat cbuf_map(shape_out[2], shape_out[3], CV_8UC1,
                     reinterpret_cast<unsigned char *>(cbuf));
    cv::Mat pred_map(shape_out[2], shape_out[3], CV_32F,
                     reinterpret_cast<float *>(pred));

    const double threshold = double(Config["det_db_thresh"]) * 255;
    const double max_value = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, max_value, cv::THRESH_BINARY);
    if (det_db_use_dilate == 1) {
        cv::Mat dilation_map;
        cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, dilation_map, dila_ele);
        bit_map = dilation_map;
    }
    auto boxes = BoxesFromBitmap(pred_map, bit_map, Config);

    std::vector<std::vector<std::vector<int>>> filter_boxes =
            FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], srcimg);

    auto img_vis = Visualization(srcimg, boxes);
    std::cout << boxes.size() << " bboxes have detected:" << std::endl;

    for (int i=0; i<boxes.size(); i++){
        std::cout << "The " << i << " box:" << std::endl;
        for (int j=0; j<4; j++){
            for (int k=0; k<2; k++){
                std::cout << boxes[i][j][k] << "\t";
            }
        }
        std::cout << std::endl;
    }

    return 0;
}
