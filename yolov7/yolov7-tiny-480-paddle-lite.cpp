#include <iostream>
#include "paddle_api.h"
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace paddle::lite_api;

int64_t ShapeProduction(const shape_t& shape) {
    int64_t res = 1;
    for (auto i : shape) res *= i;
    return res;
}

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}

int test_yolov7_tiny_480_paddle_lite() {
    std::string model_file("/Users/yang/CLionProjects/test_paddle_lite/yolov7/yolov7-tiny-x86-480-coco.nb");
    std::string image_file("/Users/yang/CLionProjects/test_paddle_lite/data/images/img.jpg");

    cv::Mat image = cv::imread(image_file, cv::IMREAD_COLOR);
    cv::Mat image_resize;
    cv::resize(image, image_resize, cv::Size(480, 480));
    image_resize.convertTo(image_resize, CV_32F, 1.0/255.0);
    float w_scale = (float)image.cols / (float)480;
    float h_scale = (float)image.rows / (float)480;

//    cv::transposeND(image, {2,0,1}, image_t);
//    cv::imshow("", image);
//    cv::waitKey(0);

//     1. Set MobileConfig
    MobileConfig config;
    // 2. Set the path to the model generated by opt tools
    config.set_model_from_file(model_file);
    // 3. Create PaddlePredictor by MobileConfig
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
    input_tensor->Resize({1, 3, 480, 480});

    auto* data = input_tensor->mutable_data<float>();

//    for (int i = 0; i < ShapeProduction(input_tensor->shape()); ++i) {
//        data[i] = 1;
//    }

    cv::Mat image_channels[3];

    cv::split(image_resize, image_channels);
    for (int j = 0; j < 3; j++) {
        memcpy(data + 480*480 * j, image_channels[j].data,480*480 * sizeof(float));
    }
//
    predictor->Run();
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
    // 转化为数据
    const float* output=output_tensor->data<float>();

//    shape_t a = output_tensor->shape();
//    for(int i = 0; i < a.size(); i++){
//        std::cout << a[i] << std::endl;
//    }

    std::vector<BoxInfo> boxes;
    for(int i = 0; i < 14175; i++){

        if(*(output+i*85+4) > 0.2){
            int cur_label = 0;
            float score = *(output+i*85+4+1);
            for (int label = 0; label < 80; label++)
            {
                //LOGD("decode_infer label %d",label);
                //LOGD("decode_infer score %f",scores[label]);
                if (*(output+i*85+5+label) > score)
                {
                    score = *(output+i*85+5+label);
                    cur_label = label;
                }
            }

//            float x = *(output+i*85+0)* 480.0f * w_scale;
            float x = *(output+i*85+0)*  w_scale;
//            float y = *(output+i*85+1)* 480.0f * h_scale;
            float y = *(output+i*85+1)* h_scale;
//            float w = *(output+i*85+2)* 480.0f * w_scale;
            float w = *(output+i*85+2)*  w_scale;
//            float h = *(output+i*85+3)* 480.0f * h_scale;
            float h = *(output+i*85+3)*  h_scale;

            boxes.push_back(BoxInfo{
                    (float)std::max(0.0, x-w/2.0),
                    (float)std::max(0.0, y-h/2.0),
                    (float)std::min((float)image.cols, (float)(x+w/2.0)),
                    (float)std::min((float)image.rows, (float)(y+h/2.0)),
                    *(output+i*85+4),
                    cur_label
            });
//            std::cout << " x1: " << (float)std::max(0.0, x-w/2.0) <<
//            " y1: " << (float)std::max(0.0, y-h/2.0) <<
//            " x2: " << (float)std::min(320.0, x+w/2.0) <<
//            " y2: " << (float)std::min(320.0, y+h/2.0) <<
//            " socre: " << *(output+i*85+4) <<
//            " label: " << cur_label << std::endl;
        }
    }

    nms(boxes, 0.6);
    for(auto &box: boxes){
        std::cout << " x1: " << box.x1 <<
                  " y1: " << box.y1 <<
                  " x2: " << box.x2 <<
                  " y2: " << box.y2 <<
                  " socre: " << box.score <<
                  " label: " << box.label << std::endl;
    }
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);

//
//
    return 0;
}
