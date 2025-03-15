#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <nlohmann/json.hpp>
#include "fast-cpp-csv-parser/csv.h"
#include <Eigen/Dense>

using namespace std;
using json = nlohmann::json;
using namespace Eigen;

struct SensorObject {
    string sensor_id;   // Camera ID (e.g., "cam_149")
    double timestamp;   // Time when object was detected
    Vector2d position;  // [x, y] coordinates
};

struct IMUData {
    double timestamp; // Time of measurement
    double heading;   // Direction of motion (angle)
    string state;     // Vehicle status (e.g., moving, stopped)
};

struct Cluster {
    int f_id;                      // Unique cluster ID
    vector<Vector2d> cluster_data; // List of objects in the cluster ([x, y, sensor_id])
    double timestamp;              // Timestamp of the cluster
};

class KalmanFilter {
public:
    KalmanFilter() {
        // State Vector [x, y, vx, vy] (position & velocity)
        x = Vector4d::Zero();
        
        // State Covariance Matrix
        P = Matrix4d::Identity() * 1000;
        
        // State Transition Matrix (Constant Velocity Model)
        F = Matrix4d::Identity();
        
        // Measurement Matrix (We only observe [x, y])
        H = Matrix<double, 2, 4>::Zero();
        H(0, 0) = H(1, 1) = 1;
        
        // Measurement Covariance Matrix (Assumed Sensor Noise)
        R = Matrix2d::Identity() * 10;
        
        // Process Noise Covariance
        Q = Matrix4d::Identity() * 0.01;
    }

    void predict(double dt) {
        // Update F with time delta
        F(0, 2) = dt;
        F(1, 3) = dt;
        
        // Predict next state
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const Vector2d& z) {
        Vector2d y = z - H * x;  // Measurement Residual
        Matrix2d S = H * P * H.transpose() + R;
        Matrix<double, 4, 2> K = P * H.transpose() * S.inverse(); // Kalman Gain

        // Update state
        x = x + K * y;
        P = (Matrix4d::Identity() - K * H) * P;
    }

    Vector2d get_position() { return x.head(2); }
    void initialize(const Vector2d& z) { x.head(2) = z; }

private:
    Vector4d x;
    Matrix4d P, F, Q;
    Matrix<double, 2, 4> H;
    Matrix2d R;
};

vector<SensorObject> read_json_data(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error: Failed to open file: " << filename << endl;
        return {};
    }

    json data;
    try { file >> data; }
    catch (const json::parse_error& e) {
        cerr << "JSON Parsing Error: " << e.what() << endl;
        return {};
    }

    vector<SensorObject> sensor_objects;
    for (const auto& item : data) {
        if (!item.begin().value().contains("timestamp") || !item.begin().value().contains("object_positions_x_y"))
            continue;

        try {
            SensorObject obj;
            obj.sensor_id = item.begin().key();
            obj.timestamp = stod(item.begin().value()["timestamp"].get<string>());
            
            if (!item.begin().value()["object_positions_x_y"].is_array() || item.begin().value()["object_positions_x_y"].empty()) {
                continue;
            }
            
            vector<double> pos = item.begin().value()["object_positions_x_y"][0].get<vector<double>>();
            obj.position = Vector2d(pos[0], pos[1]);
            
            sensor_objects.push_back(obj);
        } catch (const exception& e) {
            cerr << "Error processing entry: " << e.what() << endl;
        }
    }
    return sensor_objects;
}

vector<IMUData> read_imu_data(const string& filename) {
    io::CSVReader<5> in(filename);
    in.read_header(io::ignore_extra_column, "timestamp", "ID", "yaw", "heading", "state");
    vector<IMUData> imu_data;
    string timestamp, state;
    int id;
    double yaw, heading;
    while (in.read_row(timestamp, id, yaw, heading, state)) {
        imu_data.push_back({stod(timestamp), heading, state});
    }
    return imu_data;
}

vector<Cluster> track_objects(vector<SensorObject>& objects) {
    unordered_map<string, KalmanFilter> trackers;
    vector<Cluster> clusters;
    int cluster_id = 1;
    
    for (auto& obj : objects) {
        if (trackers.find(obj.sensor_id) == trackers.end()) {
            trackers[obj.sensor_id].initialize(obj.position);
        }
        trackers[obj.sensor_id].predict(0.1);
        trackers[obj.sensor_id].update(obj.position);
        
        Vector2d est_pos = trackers[obj.sensor_id].get_position();
        clusters.push_back({cluster_id++, {est_pos}, obj.timestamp});
    }
    return clusters;
}

void write_fused_data(const string& filename, vector<Cluster>& clusters, vector<IMUData>& imu_data) {
    ofstream file(filename);
    file << "f_timestamp,f_id,cluster_data,heading,status\n";
    for (auto& cluster : clusters) {
        double heading = 0.0;
        string status;
        for (auto& imu : imu_data) {
            if (abs(cluster.timestamp - imu.timestamp) < 1.0) {
                heading = imu.heading;
                status = imu.state;
                break;
            }
        }
        file << cluster.timestamp << "," << cluster.f_id << "," << cluster.cluster_data.size() << "," << heading << "," << status << "\n";
    }
}

int main() {
    // Step 1: Read sensor data (JSON)
    vector<SensorObject> sensor_objects = read_json_data("task_cam_data.json");
    // Step 2: Read IMU data (CSV)
    vector<IMUData> imu_data = read_imu_data("task_imu.csv");
	// Step 3: Perform clustering
	vector<Cluster> clusters = track_objects(sensor_objects);
	// Step 4: Write fused data to CSV
    write_fused_data("fused_data.csv", clusters, imu_data);
    cout << "Fusion complete. Output saved to fused_data.csv" << endl;
    return 0;
}
