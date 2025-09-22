#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fmt/format.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <H5Cpp.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <memory>

namespace fs = std::filesystem;

DEFINE_string(megadepth_path, "", "Path to MegaDepth dataset");

using namespace Eigen;
using namespace std;

// 自定义异常类
class Hdf5Exception : public std::runtime_error {
public:
    Hdf5Exception(const std::string& msg) : std::runtime_error(msg) {}
};

// HDF5文件操作辅助类
class Hdf5Reader {
public:
    Hdf5Reader(const fs::path& filename) {
        file_.openFile(filename.string(), H5F_ACC_RDONLY);
    }

    template <typename T>
    Matrix<T, Dynamic, 1> readVector(const std::string& dataset_name) {
        hid_t dataset = H5Dopen(file_.getId(), dataset_name.c_str(), H5P_DEFAULT);
        if (dataset < 0) {
            throw Hdf5Exception(fmt::format("Failed to open dataset {}", dataset_name));
        }

        hid_t dataspace = H5Dget_space(dataset);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dataspace, dims, NULL);

        Matrix<T, Dynamic, 1> result(dims[0]);
        H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, result.data());

        H5Sclose(dataspace);
        H5Dclose(dataset);
        return result;
    }

private:
    H5::H5File file_;
};

// 缓存管理类
class MegadepthCacheManager {
public:
    MegadepthCacheManager(const fs::path& cache_dir) : cache_dir_(cache_dir) {
        if (!fs::exists(cache_dir_)) {
            fs::create_directories(cache_dir_);
        }
    }

    bool exists(const std::string& key) const {
        return fs::exists(cache_dir_ / (key + ".json"));
    }

    void save(const std::string& key, const Json::Value& data) {
        std::ofstream ofs(cache_dir_ / (key + ".json"));
        ofs << data;
    }

    Json::Value load(const std::string& key) {
        std::ifstream ifs(cache_dir_ / (key + ".json"));
        Json::Value data;
        ifs >> data;
        return data;
    }

private:
    fs::path cache_dir_;
};

// 数据记录器
class MegadepthRecorder {
public:
    void record(float ADI, float PDI, float SDI) {
        // 记录到不同类别
        auto addEntry = [&](const std::string& type, float value) {
            if (value < 0.2f) {
                counts_[type]["low"]++;
                entries_[type]["Low"].push_back(value);
            } else if (value < 0.4f) {
                counts_[type]["medium"]++;
                entries_[type]["Medium"].push_back(value);
            } else if (value < 0.6f) {
                counts_[type]["high"]++;
                entries_[type]["High"].push_back(value);
            } else {
                counts_[type]["ultra"]++;
                entries_[type]["Ultra"].push_back(value);
            }
        };

        addEntry("ADI", ADI);
        addEntry("PDI", PDI);
        addEntry("SDI", SDI);
    }

    void save(const fs::path& output_path) {
        std::ofstream ofs(output_path);
        ofs << entries_;
    }

private:
    Json::Value counts_;
    Json::Value entries_;
};

// 数据集生成器
class NewDatasetGenerator {
public:
    NewDatasetGenerator(const fs::path& megadepth_path)
        : dataset_path_(megadepth_path), cache_manager_(megadepth_path / ".cache") {
        loadDatasetJson();
    }

    void run() {
        // 并行处理场景
        vector<thread> threads;
        for (const auto& scene_id : scenes_) {
            threads.emplace_back(&NewDatasetGenerator::processScene, this, scene_id);
        }

        for (auto& t : threads) {
            t.join();
        }

        // 保存统计信息
        recorder_.save(dataset_path_ / "statistics.json");
    }

    void processScene(const string& scene_id) {
        const auto& scene = scenes_[scene_id];
        const auto& tuples = scene["tuples"];

        for (size_t i = 0; i < tuples.size(); ++i) {
            try {
                processTuple(scene_id, i);
            } catch (const exception& e) {
                LOG(ERROR) << "Error processing tuple " << i << " in scene " << scene_id << ": " << e.what();
            }
        }
    }

private:
    void loadDatasetJson() {
        ifstream ifs(dataset_path_ / "dataset.json");
        Json::Value root;
        ifs >> root;

        for (const auto& item : root) {
            scenes_[item.asString()] = item;
        }
    }

    void processTuple(const string& scene_id, size_t tuple_idx) {
        const auto& scene = scenes_[scene_id];
        const auto& tuples = scene["tuples"];
        const auto& tuple = tuples[tuple_idx];

        // 获取图像路径
        string image_dir = scene["image_path"].asString();
        string depth_dir = scene["depth_path"].asString();
        string calib_dir = scene["calib_path"].asString();
        const auto& images = scene["images"];

        vector<cv::Mat> imgs;
        vector<MatrixXf> depths;
        vector<Matrix3f> Ks;
        vector<Matrix3f> Rs;
        vector<Vector3f> Ts;

        for (const auto& img_idx : tuple) {
            // 读取图像
            auto img_path = dataset_path_ / image_dir / images[img_idx].asString();
            cv::Mat img = cv::imread(img_path.string());
            if (img.empty()) {
                throw runtime_error(fmt::format("Failed to load image {}", img_path.string()));
            }
            imgs.push_back(img);

            // 读取深度图
            auto depth_path = dataset_path_ / depth_dir / (images[img_idx].asString().substr(0, images[img_idx].asString().find_last_of('.')) + ".h5");
            Hdf5Reader reader(depth_path);
            MatrixXf depth = reader.readVector<float>("depth").reshaped(img.rows, img.cols);
            depths.push_back(depth);

            // 读取标定信息
            auto calib_path = dataset_path_ / calib_dir / ("calibration_" + images[img_idx].asString() + ".h5");
            Hdf5Reader calib_reader(calib_path);
            Ks.push_back(calib_reader.readVector<float>("K").reshaped(3, 3));
            Rs.push_back(calib_reader.readVector<float>("R").reshaped(3, 3));
            Ts.push_back(calib_reader.readVector<float>("T"));
        }

        // 处理图像对
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i + 1; j < 3; ++j) {
                processImagePair(scene_id, tuple_idx, i, j, imgs[i], depths[i], Ks[i], Rs[i], Ts[i],
                                 imgs[j], depths[j], Ks[j], Rs[j], Ts[j]);
            }
        }
    }

    void processImagePair(const string& scene_id, size_t tuple_idx, size_t idx1, size_t idx2,
                          const cv::Mat& img1, const MatrixXf& depth1, const Matrix3f& K1,
                          const Matrix3f& R1, const Vector3f& T1,
                          const cv::Mat& img2, const MatrixXf& depth2, const Matrix3f& K2,
                          const Matrix3f& R2, const Vector3f& T2) {
        // 生成缓存键
        string cache_key = fmt::format("{}_{}_{}", scene_id,
                                      scene["images"][tuple[idx1]].asString().substr(0, -4),
                                      scene["images"][tuple[idx2]].asString().substr(0, -4));

        if (cache_manager_.exists(cache_key)) {
            LOG(INFO) << "Cache hit for " << cache_key;
            return;
        }

        // 计算指标
        auto [ADI, PDI, SDI] = computeDelta(img1, depth1, K1, R1, T1, img2, depth2, K2, R2, T2);

        // 保存结果
        Json::Value cache_data;
        cache_data["ADI"] = ADI;
        cache_data["PDI"] = PDI;
        cache_data["SDI"] = SDI;
        cache_manager_.save(cache_key, cache_data);

        recorder_.record(ADI, PDI, SDI);
    }

    tuple<float, float, float> computeDelta(const cv::Mat& img1, const MatrixXf& depth1, const Matrix3f& K1,
                                           const Matrix3f& R1, const Vector3f& T1,
                                           const cv::Mat& img2, const MatrixXf& depth2, const Matrix3f& K2,
                                           const Matrix3f& R2, const Vector3f& T2) {
        // 使用网格采样获取点
        auto [points1, points2, space_points, depth1_vals, depth2_vals] = getGridPoints(img1, depth1, K1, R1, T1,
                                                                                     img2, depth2, K2, R2, T2);

        // 计算角度差异
        Vector3f point_mean = space_points.rowwise().mean();
        float angle_delta = getAngleCos(point_mean, T1, T2);

        // 计算像素差异
        float depth_mean1 = depth1_vals.mean();
        float depth_mean2 = depth2_vals.mean();
        float pixel_delta = getPixelDelta(depth_mean1, K1, depth_mean2, K2);

        float ADI = angle_delta;
        float PDI = pixel_delta;
        float SDI = 0.5f * ADI + 0.5f * PDI;

        return {ADI, PDI, SDI};
    }

    tuple<MatrixXf, MatrixXf, MatrixXf, VectorXf, VectorXf> getGridPoints(
        const cv::Mat& img1, const MatrixXf& depth1, const Matrix3f& K1,
        const Matrix3f& R1, const Vector3f& T1,
        const cv::Mat& img2, const MatrixXf& depth2, const Matrix3f& K2,
        const Matrix3f& R2, const Vector3f& T2) {
        // 网格采样
        int step = 8;
        MatrixX2f points1;
        points1.resize(img1.rows/step * img1.cols/step, 2);

        int idx = 0;
        for (int y = step; y < img1.rows; y += step) {
            for (int x = step; x < img1.cols; x += step) {
                if (depth1(y, x) > 0) {
                    points1.row(idx++) << x, y;
                }
            }
        }
        points1.conservativeResize(idx, 2);

        // 转换到齐次坐标
        MatrixX3f homo_points1 = MatrixX3f::Zero(points1.rows(), 3);
        homo_points1 << points1, VectorXf::Ones(points1.rows());

        // 转换到空间坐标
        MatrixX3f space_points = R1.inverse() * (K1.inverse() * (depth1.array() * homo_points1.array()).matrix().transpose() - T1.replicate(1, homo_points1.rows()));
        space_points = space_points.transpose();

        // 投影到第二张图像
        MatrixX3f homo_points2 = K2 * (R2 * space_points.transpose() + T1.replicate(1, space_points.rows()));
        homo_points2 = homo_points2.array().rowwise() / homo_points2.array().col(2);
        MatrixX2f points2 = homo_points2.leftCols<2>();

        // 筛选有效点
        VectorXf valid_mask = (points2.array().col(0) >= 0) & (points2.array().col(0) < img2.cols) &
                             (points2.array().col(1) >= 0) & (points2.array().col(1) < img2.rows);
        VectorXf depth1_vals = depth1.array().select(0, 1).cast<float>();
        VectorXf depth2_vals = depth2.array().select(0, 1).cast<float>();

        return {points1, points2, space_points, depth1_vals, depth2_vals};
    }

    float getAngleCos(const Vector3f& A, const Vector3f& B, const Vector3f& C) {
        float a = (B - C).norm();
        float b = (A - C).norm();
        float c = (A - B).norm();
        return acos((b*b + c*c - a*a) / (2*b*c));
    }

    float getPixelDelta(float depth_mean1, const Matrix3f& K1, float depth_mean2, const Matrix3f& K2) {
        float fx1 = K1(0, 0);
        float fy1 = K1(1, 1);
        float fx2 = K2(0, 0);
        float fy2 = K2(1, 1);

        float theta = M_PI_4;
        float fp1 = (1.0f/depth_mean1) * sqrt(fx1*fx1 * cos(theta)*cos(theta) + fy1*fy1 * sin(theta)*sin(theta));
        float fp2 = (1.0f/depth_mean2) * sqrt(fx2*fx2 * cos(theta)*cos(theta) + fy2*fy2 * sin(theta)*sin(theta));
        return abs(fp1 - fp2) / (fp1 + fp2);
    }

    fs::path dataset_path_;
    map<string, Json::Value> scenes_;
    MegadepthCacheManager cache_manager_;
    MegadepthRecorder recorder_;
};

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    if (FLAGS_megadepth_path.empty()) {
        LOG(ERROR) << "Please specify --megadepth_path";
        return 1;
    }

    NewDatasetGenerator generator(FLAGS_megadepth_path);
    generator.run();

    return 0;
}