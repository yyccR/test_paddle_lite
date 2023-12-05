
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

    float *dout_c0 = dout;
    float *dout_c1 = dout + size;
    float *dout_c2 = dout + size * 2;

    for (int i = 0; i < size; i++) {
        dout_c0[i] = (din[i * 3] - mean[0]) * scale[0];
        dout_c1[i] = (din[i * 3 + 1] - mean[1]) * scale[1];
        dout_c2[i] = (din[i * 3 + 2] - mean[2]) * scale[2];
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

std::vector<std::vector<std::vector<int>>>
RunDetModel(std::shared_ptr<PaddlePredictor> predictor, cv::Mat img,
            std::map<std::string, double> Config, std::vector<double> *times) {
    // Read img
    int max_side_len = int(Config["max_side_len"]);
    int det_db_use_dilate = int(Config["det_db_use_dilate"]);

    cv::Mat srcimg;
    img.copyTo(srcimg);

    auto preprocess_start = std::chrono::steady_clock::now();
    std::vector<float> ratio_hw;
    img = DetResizeImg(img, max_side_len, ratio_hw);
    cv::Mat img_fp;
    img.convertTo(img_fp, CV_32FC3, 1.0 / 255.f);

    // Prepare input data from image
    std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
    input_tensor0->Resize({1, 3, img_fp.rows, img_fp.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    const float *dimg = reinterpret_cast<const float *>(img_fp.data);
//    NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
    MeanScale(dimg, data0, img_fp.rows * img_fp.cols, mean, scale);
    auto preprocess_end = std::chrono::steady_clock::now();

    // Run predictor
    auto inference_start = std::chrono::steady_clock::now();
    predictor->Run();

    // Get output and post process
    std::unique_ptr<const Tensor> output_tensor(
            std::move(predictor->GetOutput(0)));
    auto *outptr = output_tensor->data<float>();
    auto shape_out = output_tensor->shape();
    auto inference_end = std::chrono::steady_clock::now();

    // Save output
    auto postprocess_start = std::chrono::steady_clock::now();
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
    auto postprocess_end = std::chrono::steady_clock::now();

    std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
    times->push_back(double(preprocess_diff.count() * 1000));
    std::chrono::duration<float> inference_diff = inference_end - inference_start;
    times->push_back(double(inference_diff.count() * 1000));
    std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
    times->push_back(double(postprocess_diff.count() * 1000));

    return filter_boxes;
}

cv::Mat RunClsModel(cv::Mat img, std::shared_ptr<PaddlePredictor> predictor_cls,
                    const float thresh = 0.9) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat srcimg;
    img.copyTo(srcimg);
    cv::Mat crop_img;
    img.copyTo(crop_img);
    cv::Mat resize_img;

    int index = 0;
    float wh_ratio =
            static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

    resize_img = ClsResizeImg(crop_img);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);

    const float *dimg = reinterpret_cast<const float *>(resize_img.data);

    std::unique_ptr<Tensor> input_tensor0(std::move(predictor_cls->GetInput(0)));
    input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
    auto *data0 = input_tensor0->mutable_data<float>();

//    NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
    MeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
    // Run CLS predictor
    predictor_cls->Run();

    // Get output and run postprocess
    std::unique_ptr<const Tensor> softmax_out(
            std::move(predictor_cls->GetOutput(0)));
    auto *softmax_scores = softmax_out->mutable_data<float>();
    auto softmax_out_shape = softmax_out->shape();
    float score = 0;
    int label = 0;
    for (int i = 0; i < softmax_out_shape[1]; i++) {
        if (softmax_scores[i] > score) {
            score = softmax_scores[i];
            label = i;
        }
    }
    if (label % 2 == 1 && score > thresh) {
        cv::rotate(srcimg, srcimg, 1);
    }
    return srcimg;
}

void RunRecModel(std::vector<std::vector<std::vector<int>>> boxes, cv::Mat img,
                 std::shared_ptr<PaddlePredictor> predictor_crnn,
                 std::vector<std::string> &rec_text,
                 std::vector<float> &rec_text_score,
                 std::vector<std::string> charactor_dict,
                 std::shared_ptr<PaddlePredictor> predictor_cls,
                 int use_direction_classify,
                 std::vector<double> *times,
                 int rec_image_height) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

    cv::Mat srcimg;
    img.copyTo(srcimg);
    cv::Mat crop_img;
    cv::Mat resize_img;

    int index = 0;

    std::vector<double> time_info = {0, 0, 0};
    for (int i = boxes.size() - 1; i >= 0; i--) {
        std::cout << "boxes.size(): " << boxes.size() << " cur i: " << i << std::endl;
//        auto preprocess_start = std::chrono::steady_clock::now();
        crop_img = GetRotateCropImage(srcimg, boxes[i]);
//        std::string file_name("/Users/yang/CLionProjects/test_paddle_lite/data/images/");
//        file_name += std::to_string(i) + "crop_img.jpg";
//        cv::imwrite(file_name, crop_img);
        std::cout << " crop_img.cols: " << crop_img.cols << " crop_img.rows: " << crop_img.rows << std::endl;

        if (use_direction_classify >= 1) {
            crop_img = RunClsModel(crop_img, predictor_cls);
        }
        float wh_ratio =
                static_cast<float>(crop_img.cols) / static_cast<float>(crop_img.rows);

        resize_img = CrnnResizeImg(crop_img, wh_ratio, rec_image_height);
//        std::string file_name("/Users/yang/CLionProjects/test_paddle_lite/data/images/");
//        file_name += std::to_string(i) + "i_resize_img.jpg";
//        cv::imwrite(file_name, resize_img);
        resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.f);
        std::cout << " resize_img.cols: " << resize_img.cols << " resize_img.rows: " << resize_img.rows << std::endl;

        const float *dimg = reinterpret_cast<const float *>(resize_img.data);

        std::unique_ptr<Tensor> input_tensor0(
                std::move(predictor_crnn->GetInput(0)));
        input_tensor0->Resize({1, 3, resize_img.rows, resize_img.cols});
        auto *data0 = input_tensor0->mutable_data<float>();
//        std::cout << " input_tensor0: " << input_tensor0->shape()[0] << " " << input_tensor0->shape()[1] << " " << input_tensor0->shape()[2] << " " << input_tensor0->shape()[3] << std::endl;


//        NeonMeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
        MeanScale(dimg, data0, resize_img.rows * resize_img.cols, mean, scale);
        std::cout << " data0: " << data0[0] << " " << data0[1] << " " << data0[2] << " " << data0[3] << std::endl;

        auto preprocess_end = std::chrono::steady_clock::now();
        //// Run CRNN predictor
        auto inference_start = std::chrono::steady_clock::now();
        predictor_crnn->Run();
        std::cout << "  predictor_crnn->Run(); " << std::endl;
        // Get output and run postprocess
        std::unique_ptr<const Tensor> output_tensor0(
                std::move(predictor_crnn->GetOutput(0)));
        auto *predict_batch = output_tensor0->data<float>();
        auto predict_shape = output_tensor0->shape();
        auto inference_end = std::chrono::steady_clock::now();

        // ctc decode
        auto postprocess_start = std::chrono::steady_clock::now();
        std::string str_res;
        int argmax_idx;
        int last_index = 0;
        float score = 0.f;
        int count = 0;
        float max_value = 0.0f;

        for (int n = 0; n < predict_shape[1]; n++) {
//            std::cout << " predict_shape[1]: " << predict_shape[1] << " n: " << n << std::endl;

            argmax_idx = int(Argmax(&predict_batch[n * predict_shape[2]],
                                    &predict_batch[(n + 1) * predict_shape[2]]));
            max_value =
                    float(*std::max_element(&predict_batch[n * predict_shape[2]],
                                            &predict_batch[(n + 1) * predict_shape[2]]));
            if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
//                std::cout << charactor_dict.size() << " " << argmax_idx << " " << charactor_dict[argmax_idx] << std::endl;
                str_res += charactor_dict[argmax_idx];
            }
            last_index = argmax_idx;
        }
//        std::cout << "score: " << score << " count: " << count << std::endl;
        if(count){
            score /= count;
        }
        rec_text.push_back(str_res);
        rec_text_score.push_back(score);
        auto postprocess_end = std::chrono::steady_clock::now();

//        std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
//        time_info[0] += double(preprocess_diff.count() * 1000);
//        std::chrono::duration<float> inference_diff = inference_end - inference_start;
//        time_info[1] += double(inference_diff.count() * 1000);
//        std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
//        time_info[2] += double(postprocess_diff.count() * 1000);

    }

    times->push_back(time_info[0]);
    times->push_back(time_info[1]);
    times->push_back(time_info[2]);
}


int test_paddle_ocr_v4_paddle_lite() {
    std::string det_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_PP-OCRv4_det_opt.nb");
    std::string rec_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_PP-OCRv4_rec_opt.nb");
    std::string cls_model_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/v4/ch_ppocr_mobile_v2.0_cls_opt.nb");
    std::string ppocr_keys_v1_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/paddle_ocr_lib/ppocr_keys_v1.txt");
    std::string config_file("/Users/yang/CLionProjects/test_paddle_lite/paddle_ocr/paddle_ocr_lib/config.txt");
    std::string image_file("/Users/yang/CLionProjects/test_paddle_lite/data/images/insurance.png");

    auto Config = LoadConfigTxt(config_file);
    int use_direction_classify = int(Config["use_direction_classify"]);
    int rec_image_height = int(Config["rec_image_height"]);
    auto charactor_dict = ReadDict(ppocr_keys_v1_file);
    charactor_dict.insert(charactor_dict.begin(), "#"); // blank char for ctc
    charactor_dict.push_back(" ");

    cv::Mat img = cv::imread(image_file, cv::IMREAD_COLOR);

    //// load det
    MobileConfig det_config;
    det_config.set_model_from_file(det_model_file);
    std::shared_ptr<PaddlePredictor> det_predictor = CreatePaddlePredictor<MobileConfig>(det_config);
    std::cout << "load det model" << std::endl;

    //// load rec
    MobileConfig rec_config;
    rec_config.set_model_from_file(rec_model_file);
    std::shared_ptr<PaddlePredictor> rec_predictor = CreatePaddlePredictor<MobileConfig>(rec_config);
    std::cout << "load rec model" << std::endl;

    //// load cls
    MobileConfig cls_config;
    cls_config.set_model_from_file(cls_model_file);
    std::shared_ptr<PaddlePredictor> cls_predictor = CreatePaddlePredictor<MobileConfig>(cls_config);
    std::cout << "load cls model" << std::endl;


    //// 推理
    std::vector<double> det_times;
    auto boxes = RunDetModel(det_predictor, img, Config, &det_times);
    std::cout << "det size: " << boxes.size() << std::endl;
    std::vector<std::string> rec_text;
    std::vector<float> rec_text_score;
    std::vector<double> rec_times;
    RunRecModel(boxes, img, rec_predictor, rec_text, rec_text_score,
                charactor_dict, cls_predictor, use_direction_classify, &rec_times, rec_image_height);
    std::cout << "rec size: " << rec_text.size() << std::endl;


    auto img_vis = Visualization(img, boxes);

    //// print recognized text
    for (int i = 0; i < rec_text.size(); i++) {
        std::cout << i << "\t" << rec_text[i] << "\t" << rec_text_score[i]
                  <<  std::endl;

    }

    return 0;
}
