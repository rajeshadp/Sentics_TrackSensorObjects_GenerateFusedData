#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <nlohmann/json.hpp>
#include "fast-cpp-csv-parser/csv.h"
#include <Eigen/Dense>
#include <sstream>
#include <ctime>
#include <chrono>

using namespace std;
using json = nlohmann::json;
using namespace Eigen;

// Function to convert "YYYY-MM-DD HH:MM:SS.sss" to a timestamp (seconds) and also T inside timestamp of IMU csv
double parseTimestamp(const std::string& timestamp) {
    std::tm tm = {};
    std::string processed_timestamp = timestamp;

    // Replace 'T' with ' ' to match expected format
    size_t t_pos = processed_timestamp.find('T');
    if (t_pos != std::string::npos) {
        processed_timestamp[t_pos] = ' '; 
    }

    // Parse date and time (ignoring fractional seconds for now)
    std::istringstream ss(processed_timestamp);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    
    if (ss.fail()) {
        std::cerr << "Error parsing timestamp: " << timestamp << std::endl;
        return 0.0; // Handle failure case
    }

    // Extract microseconds
    size_t dot_pos = processed_timestamp.find_last_of('.');
    double fraction = 0.0;
    if (dot_pos != std::string::npos) {
        std::string fraction_str = processed_timestamp.substr(dot_pos);
        fraction = std::stod(fraction_str);
    }

    // Convert to time_t (epoch time) and add fraction
    std::time_t time_epoch = std::mktime(&tm);
    return static_cast<double>(time_epoch) + fraction;
}
struct SensorObject {
    string sensor_id;   // Camera ID (e.g., "cam_149")
    double timestamp;   // Time when object was detected
    string timestamp_str;   // Time when object was detected
    Vector2d position;  // [x, y] coordinates
};

struct IMUData {
    double timestamp; // Time of measurement
    double heading;   // Direction of motion (angle)
    string state;     // Vehicle status (e.g., moving, stopped)
};

/* a cluster represents a group of detected objects or positions 
   that belong to the same tracked entity. */
struct Cluster {
    int f_id;                      // Unique cluster ID
    vector<Vector2d> cluster_data; // List of objects in the cluster ([x, y, sensor_id])
    double timestamp;              // Timestamp of the cluster
    string timestamp_str;

    // Function to format cluster data as a string
    string format_cluster_data() const {
        stringstream ss;
        ss << "[";
        for (size_t i = 0; i < cluster_data.size(); i++) {
            ss << "(" << cluster_data[i].x() << "," << cluster_data[i].y() << ")";
            if (i != cluster_data.size() - 1) ss << "; ";  // Separate multiple points
        }
        ss << "]";
        return ss.str();
    }
};

// Kalman Filter for Heading Tracking
/*
	Kalman Filter = Predict, Correct, Store

	1.Prediction is a guess based on motion.
	2.Update corrects that guess using sensor data.
	3.Clustering organizes and stores the final best estimates.
*/

class KalmanFilterHeading {
public:
    KalmanFilterHeading() {
        x = Vector2d::Zero(); // State vector: [heading, heading_rate]
        P = Matrix2d::Identity() * 1000; 
        F = Matrix2d::Identity(); 
        H = Matrix<double, 1, 2>::Zero();
        H(0, 0) = 1;  
        R = Matrix<double, 1, 1>::Identity() * 5;  
        Q = Matrix2d::Identity() * 0.01; 
    }

    void predict(double dt) {
        F(0, 1) = dt; // Heading update using rate of change
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(double z) {
        Vector<double, 1> y;
        y(0) = z - (H * x)(0); 
        Matrix<double, 1, 1> S = H * P * H.transpose() + R;
        Matrix<double, 2, 1> K = P * H.transpose() * S.inverse();
        x = x + K * y;
        P = (Matrix2d::Identity() - K * H) * P;
    }

    double get_heading() { return x(0); }
    void initialize(double z) { x(0) = z; }

private:
    Vector2d x;    // State vector: [heading, heading_rate]
    Matrix2d P, F, Q;  // Covariance, state transition, process noise
    Matrix<double, 1, 2> H; // Measurement matrix
    Matrix<double, 1, 1> R; // Measurement noise

};

void write_fused_data(const string& filename, vector<Cluster>& clusters, vector<IMUData>& imu_data) {
    ofstream file(filename);
    file << "f_timestamp,f_id,cluster_data,heading,status\n";

    unordered_map<int, KalmanFilterHeading> heading_filters;
    for (auto& imu : imu_data) {
        int id = static_cast<int>(imu.timestamp);
        if (heading_filters.find(id) == heading_filters.end()) {
            heading_filters[id].initialize(imu.heading);
        }
        heading_filters[id].predict(0.1);
        heading_filters[id].update(imu.heading);
    }

    for (auto& cluster : clusters) {
        double filtered_heading = 0.0;
        string status;
        
        for (auto& imu : imu_data) {
            if (abs(cluster.timestamp - imu.timestamp) < 1.0) {
                int id = static_cast<int>(imu.timestamp);
                if (heading_filters.find(id) != heading_filters.end()) {
                    filtered_heading = heading_filters[id].get_heading();
                } else {
                    filtered_heading = imu.heading; 
                }
                status = imu.state;
                break;
            }
        }

        file << cluster.timestamp_str << "," << cluster.f_id << "," 
             << cluster.format_cluster_data() << "," << filtered_heading << "," << status << "\n";
    }
}

vector<IMUData> read_imu_data(const string& filename) {
    io::CSVReader<5> in(filename);
    in.read_header(io::ignore_extra_column, "timestamp", "ID", "yaw", "heading", "state");
    vector<IMUData> imu_data;
    string timestamp, state;
    int id;
    double yaw, heading;
    while (in.read_row(timestamp, id, yaw, heading, state)) {
        imu_data.push_back({parseTimestamp(timestamp), heading, state});
    }
    return imu_data;
}

/* This function tracks objects detected by sensors using a Kalman Filter,
   groups them into clusters, and returns the estimated positions. 
		Input: A vector of sensor-detected objects (objects).
		Output: A vector of Cluster objects containing estimated positions.
   */
vector<Cluster> track_objects(vector<SensorObject>& objects) {
    unordered_map<string, Vector2d> trackers;
    unordered_map<string, int> cluster_map;  
    vector<Cluster> clusters;
    int next_cluster_id = 1;

    for (auto& obj : objects) {
        if (trackers.find(obj.sensor_id) == trackers.end()) {
            trackers[obj.sensor_id] = obj.position;
            cluster_map[obj.sensor_id] = next_cluster_id++;  
        }

        clusters.push_back({cluster_map[obj.sensor_id], {trackers[obj.sensor_id]}, obj.timestamp, obj.timestamp_str});
    }
    return clusters;
}
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
            obj.sensor_id     = item.begin().key();
            obj.timestamp     = parseTimestamp(item.begin().value()["timestamp"].get<string>());
            obj.timestamp_str = item.begin().value()["timestamp"].get<string>();
            
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
