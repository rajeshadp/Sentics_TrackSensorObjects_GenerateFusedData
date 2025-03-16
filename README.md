Headers used and purpose
==========================
iostream / fstream → Handles input/output operations, including file reading/writing.

vector → Used for storing sensor objects, IMU data, and clusters.

cmath → Provides mathematical functions (not explicitly used here).

nlohmann/json.hpp → A JSON library used to parse sensor data from JSON files.

fast-cpp-csv-parser/csv.h → A high-performance CSV parser used for reading IMU data.

Eigen/Dense → Provides matrix operations for implementing the Kalman filter.
🔹 Vectors & Matrices – Supports fixed-size (Vector2d, Matrix3d) and dynamic-size (MatrixXd) operations.
🔹 Linear Algebra – Solves equations, performs matrix decomposition, and handles transformations.
🔹 Optimized Performance – Uses expression templates to avoid unnecessary memory allocations.
🔹 Header-Only – No need for separate compilation; just include <Eigen/Dense>.

This program reads sensor data, tracks objects using a Kalman filter, and fuses the data with IMU (Inertial Measurement Unit) readings to generate a refined dataset. Finally, it writes the fused data to a CSV file.

**Key Steps:**
Read Sensor Data (JSON) – Loads object positions from a sensor file.

Read IMU Data (CSV) – Loads IMU readings (e.g., heading, state).

Track Objects (Kalman Filter) – Uses a Kalman filter to smooth object positions and estimate movement.

Fuse Data – Combines object positions with IMU data based on timestamps.

Save to CSV – Writes the processed and fused data for further analysis.

This helps reduce sensor noise, track moving objects more accurately, and correlate sensor data with IMU readings for better insights.

Compile & execute:
==================
I used debian WSL on windows 11 pro, where other input files task_cam_data.json, task_imu.csv are also placed along with fusion.cpp file with business logic:

g++ -std=c++17 fusion.cpp -o fusion -I ../tempInstall/ -I /usr/include/eigen3

./fusion
