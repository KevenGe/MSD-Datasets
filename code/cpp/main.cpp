#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <filesystem>
#include <chrono>
#include <future>

//
#include <windows.h>

//
#include <fmt/format.h>
#include <glog/logging.h>
#include <json/json.h>
#include <opencv2/opencv.hpp>
#include <H5Cpp.h>

using namespace std;
namespace fs = std::filesystem;


class NewDatasetGenerator {
public:
    NewDatasetGenerator(const fs::path &megadepth_path) {
        this->megadepth_path = megadepth_path;
        this->megadepth_json_path = megadepth_path / "dataset.json";

        ifstream ifs(this->megadepth_json_path);
        Json::Reader reader;
        reader.parse(ifs, root);

        if (root.isObject()) {
            scene_ids = root.getMemberNames();
        }
    };

    void run_dataset() {
        for (int i = 0; i < scene_ids.size(); i++) {
            const string &scene_id = scene_ids[i];

            auto scene = root[scene_id];
            auto image_dir_path = scene["image_path"];
            auto depth_dir_path = scene["depth_path"];
            auto calib_dir_path = scene["calib_path"];
            auto image_names = scene["images"];
            auto tuples = scene["tuples"];

            LOG(INFO) << fmt::format("[SceneID: {}, {}/{}]", scene_id, i, scene_ids.size());

            for (int j = 0; j < tuples.size(); ++j) {
                LOG(INFO) << fmt::format("[SceneID: {}, {}/{}] [tupID: {}/{}]", scene_id, i, scene_ids.size(), j,
                                         tuples.size());

                vector<cv::Mat> images;
                vector<cv::Mat> depths;
                vector<cv::Mat> Ks;
                vector<cv::Mat> Rs;
                vector<cv::Mat> Ts;

                for (int k = 0; k < 3; k++) {
                    // images
                    auto img = cv::imread(
                        ((megadepth_path / image_dir_path.asString()) / image_names[0].asString()).string(),
                        cv::IMREAD_COLOR);
                    images.push_back(img);

                    // depths
                    H5::H5File h5;
                    h5.openFile(
                        ((megadepth_path / depth_dir_path.asString()) / image_names[0].asString().substr(
                             0, image_names[0].asString().size() - 4).append(".h5")).string(), H5F_ACC_RDONLY);
                    H5::DataSet dataset = h5.openDataSet("depth");
                    H5::DataSpace dataspace = dataset.getSpace();

                    // 长度
                    hsize_t dims[2];
                    dataspace.getSimpleExtentDims(dims, NULL);
                    int rows = static_cast<int>(dims[0]);
                    int cols = static_cast<int>(dims[1]);

                    int cv_type = CV_32F;

                    float *data_buf = new float[rows * cols];
                    dataset.read(data_buf, H5::PredType::NATIVE_FLOAT);

                    cv::Mat mat(rows, cols, CV_32F, data_buf);
                    depths.push_back(mat.clone());
                    delete []data_buf;
                    h5.close();

                    //
                    h5.openFile(
                        ((megadepth_path / calib_dir_path.asString()) / string("calibration_").
                         append(image_names[0].asString()).append(".h5")).string(), H5F_ACC_RDONLY);

                    // Ks
                    dataset = h5.openDataSet("K");
                    dataspace = dataset.getSpace();

                    dataspace.getSimpleExtentDims(dims, NULL);
                    rows = static_cast<int>(dims[0]);
                    cols = static_cast<int>(dims[1]);

                    cv_type = CV_32F;
                    float *data_buf2 = new float[rows * cols];
                    dataset.read(data_buf2, H5::PredType::NATIVE_FLOAT);
                    cv::Mat mat2(rows, cols, CV_32F, data_buf2);
                    Ks.push_back(mat2.clone());
                    delete []data_buf2;

                    // Rs
                    dataset = h5.openDataSet("R");
                    dataspace = dataset.getSpace();

                    dataspace.getSimpleExtentDims(dims, NULL);
                    rows = static_cast<int>(dims[0]);
                    cols = static_cast<int>(dims[1]);

                    cv_type = CV_32F;
                    float *data_buf3 = new float[rows * cols];
                    dataset.read(data_buf3, H5::PredType::NATIVE_FLOAT);
                    cv::Mat mat3(rows, cols, CV_32F, data_buf3);
                    Rs.push_back(mat3.clone());
                    delete []data_buf3;

                    // Ts
                    dataset = h5.openDataSet("T");
                    dataspace = dataset.getSpace();

                    dataspace.getSimpleExtentDims(dims, NULL);
                    rows = static_cast<int>(dims[0]);
                    cols = static_cast<int>(dims[1]);

                    cv_type = CV_32F;
                    float *data_buf4 = new float[rows * cols];
                    dataset.read(data_buf4, H5::PredType::NATIVE_FLOAT);
                    cv::Mat mat4(rows, cols, CV_32F, data_buf4);
                    Rs.push_back(mat4.clone());
                    delete []data_buf4;

                    h5.close();
                }
            }

            break;
        }
    };

    tuple<float, float, float> compute_delta(
        const cv::Mat &image1,
        const cv::Mat &depth1,
        const cv::Mat &K1,
        const cv::Mat &R1,
        const cv::Mat &T1,
        const cv::Mat &image2,
        const cv::Mat &depth2,
        const cv::Mat &K2,
        const cv::Mat &R2,
        const cv::Mat &T2
    ) {

        const int POINTS_RATIO = 8;

        //
        vector<cv::Point2i> points1_vec;
        for (int y = POINTS_RATIO; y < image1.rows; y += POINTS_RATIO) {

        }

        float w1 = 1.0;
        float w2 = 1.0;
        float ADI = 0;
        float PDI = 0;
        float SDI = ADI * w1 + PDI * w2;
        return make_tuple(ADI, PDI, SDI);
    }

private:
    fs::path megadepth_path;
    fs::path megadepth_json_path;
    Json::Value root;
    Json::Value::Members scene_ids;
};

int main() {
    std::cout << "Hello World!\n";

    fs::path megadepth_path = R"(D:\Datasets\MegaDepth\v2-from_disk_data_unziped\datasets.epfl.ch\disk-data\megadepth)";

    LOG(INFO) << "Hello World!";
    LOG(INFO) << fmt::format("MegaDepth Path = {}", megadepth_path.string());

    if (!fs::exists(megadepth_path)) {
        LOG(FATAL) << "No MegaDepth path exists";
        return -1;
    }

    if (!fs::is_directory(megadepth_path)) {
        LOG(FATAL) << "No MegaDepth path isn't a directory";
        return -1;
    }

    auto ndg = make_unique<NewDatasetGenerator>(megadepth_path);
    ndg->run_dataset();

    return 0;
}
