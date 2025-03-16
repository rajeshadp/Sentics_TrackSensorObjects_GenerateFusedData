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

/* a cluster represents a group of detected objects or positions 
   that belong to the same tracked entity. */
struct Cluster {
    int f_id;                      // Unique cluster ID
    vector<Vector2d> cluster_data; // List of objects in the cluster ([x, y, sensor_id])
    double timestamp;              // Timestamp of the cluster
};

/*
	Kalman Filter = Predict, Correct, Store

	1.Prediction is a guess based on motion.
	2.Update corrects that guess using sensor data.
	3.Clustering organizes and stores the final best estimates.
*/

class KalmanFilter {
public:
    KalmanFilter() {

        // State Vector [x, y, vx, vy] (Object's position & velocity)
		/*	A 4D vector that represents the object's position (x, y) and velocity (vx, vy).
		 Initial State (Zero Vector):

		The object starts with an unknown position and velocity.
		 Example Values (if an object is detected at (100, 50) moving at (5, -2) pixels/frame):
        
		x << 100, 50, 5, -2;  // [x, y, vx, vy]	 */ 

        x = Vector4d::Zero();
        
        // State Covariance Matrix  (Uncertainty in State Estimate)
      /*   Represents the uncertainty in x (position & velocity estimates).
           Diagonal values are large (1000), meaning high uncertainty initially.
           Example Values (if our position is accurate, but velocity is uncertain):
             P << 10, 0, 0, 0,
                  0, 10, 0, 0,
                  0, 0, 1000, 0,
                  0, 0, 0, 1000;  // Small uncertainty in position, large in velocity     */     

    	 P = Matrix4d::Identity() * 1000;
        
        // State Transition Matrix (Motion Model we use is Constant Velocity Model)
		/*  Defines how the state evolves over time using a constant velocity model.
            Initially set as identity, but it is updated dynamically with Δt (time step).
              Updated Example (Δt = 1s):
              F << 1, 0, 1, 0,
                   0, 1, 0, 1,
                   0, 0, 1, 0,
                   0, 0, 0, 1;  // Position updated using velocity
					This means:

					New x = Old x + vx × Δt
					New y = Old y + vy × Δt       */

        F = Matrix4d::Identity();
        
        // Measurement Matrix ((How We Observe the State) We only observe [x, y])
		/* Converts the 4D state [x, y, vx, vy] into a 2D measurement [x, y].
           This means our sensor only observes position, not velocity.
           Example:
		     H << 1, 0, 0, 0,
                  0, 1, 0, 0;  // Only extracts x, y  */

        H = Matrix<double, 2, 4>::Zero();
        H(0, 0) = H(1, 1) = 1;
        
        // Measurement Covariance Matrix (Assumed Sensor Noise)
		/*  Represents sensor measurement noise (how uncertain our position measurements are).
            A higher value (10) means noisy sensor readings.
               Example Values:
                  If our sensor is precise, we might use:
				  R << 1, 0,
                       0, 1;  // Low uncertainty
                 If our sensor is noisy, we use larger values:
				 R << 50, 0,
                       0, 50;  // High uncertainty       */

        R = Matrix2d::Identity() * 10;
        
        // Process Noise Covariance (System Uncertainty)
		/* Represents uncertainty in the motion model (errors in velocity & acceleration).
			A higher value means our model is less reliable.
			 Example Values:

			If we trust our motion model, we keep it low (0.01).
			If we expect random changes in movement, we increase it:
			Q << 0.5, 0, 0, 0,
				 0, 0.5, 0, 0,
				 0, 0, 0.1, 0,
				 0, 0, 0, 0.1;  // More uncertainty in velocity  */

        Q = Matrix4d::Identity() * 0.01;
    }

    /* Real-World Example: Car Tracking
      A car moving at vx = 10 m/s, vy = 5 m/s from position (100, 50).
      predict(1.0) estimates where the car will be in 1 second before getting the next sensor reading.
      The new estimated position is (110, 55), but uncertainty increases. */
    void predict(double dt) {
        /* // Update F with time delta i.e state transition matrix (F) models how the object's state changes over time. 
           If an object moves at vx = 10 m/s, after dt = 1s, its new position is:
                    𝑥′=𝑥+𝑣𝑥⋅𝑑𝑡     This modification ensures position updates based on velocity.
           New F Matrix (for dt = 1s)		

						​1 0 1 0
				  F  =	0 1 0 1
						0 0 1 0
						0 0 0 1
						
		The 1 values in (0,2) and (1,3) account for constant velocity motion.				
		*/ 	
        F(0, 2) = dt; // Position x depends on velocity vx over time dt
        F(1, 3) = dt; // Position y depends on velocity vy over time dt
        
        // Predict next state i.e Computes where the object should be after dt seconds using the motion model.
		/* The equation:
           𝑥′=𝐹⋅x     applies linear motion prediction. 
			Example Calculation (Before Prediction)
			Assume:	
					100
					 50
				X =	 10
					  5
           (x = 100, y = 50, vx = 10, vy = 5)					  

          Applying:
						​1 0 1 0      100     110
				 𝑥′=	0 1 0 1       50  =   55
						0 0 1 0   .   10      10
						0 0 0 1        5       5
           New predicted position: (110, 55)						
          (The object moved 10 units in x and 5 units in y.)
		   */
        x = F * x;
		
		//Predict the State Covariance (P)
		/*  P represents uncertainty in the estimated state.
           Uncertainty increases over time as we move forward without a measurement update.
           The process noise matrix Q models how much error we expect in prediction.
         Effect:
           F * P * Fᵀ propagates the previous uncertainty forward.
           Q adds a small amount of process noise (random drift).  */

        P = F * P * F.transpose() + Q;
    }

	/*  After the predict step (which estimates the new state), 
	    the update step adjusts this prediction based on a new sensor measurement (z).
	    This reduces uncertainty and aligns the estimated state with real-world observations.  */
    void update(const Vector2d& z) {
		
		/* calculates the difference between the actual sensor measurement (z) and the predicted measurement (H * x).
		The predicted state x contains estimated position (x, y, vx, vy).
		The measurement matrix H extracts only (x, y) because sensors usually don’t measure velocity directly.
		y is the error (or "innovation") between what we expected and what we actually observed.  
		Example Calculation:
			Suppose:		
					110
					 55
				x =	 10
					  5
       (Predicted position: (110, 55))
       Sensor reports:
					108
                z =	 56
		(Actual detected position: (108, 56))
        Measurement matrix H (extracting only position):	
					1 0 0 0
				H =	0 1 0 0
        Predicted measurement:				
      
					1 0 0 0    110    110
			H . x =	0 1 0 0  . 	55  =  55
								10
								 5 
        Residual (innovation)
		                   108    110     -2
            y = z - H.x =   56  -  55  =   1
			
		This tells us the predicted position was slightly off (-2, +1)		*/	 	
		
        Vector2d y = z - H * x;   
		
		/* Compute the Residual Covariance (S)
		   S represents how much uncertainty we have in this measurement comparison.
		   It combines:
		      Predicted uncertainty (P) projected into measurement space.
			  Sensor noise (R) (uncertainty in the measurement).
            Example Calculation:			 
				Let’s assume:
					1000   0    0    0
					 0    1000  0    0
      			P =	 0     0   1000  0
					 0     0    0   1000
		(Since we started with high uncertainty.)				
		And measurement noise:
							10   0
                    R =		 0  10
       Applying:
		The large 1000 values come from the high initial uncertainty (P).
        The 10 values come from sensor noise (R).
													1000   0    0    0       1  0
		S = H P transpose(H) + R =	 1 0 0 0 	.	 0    1000  0    0       0  1       10   0
									 0 1 0 0		 0     0   1000  0    .  0	0   +    0  10
													 0     0    0   1000     0  0

						1010    0
				S =		  0   1010
        This matrix tells us the total uncertainty in the measurement in both the x and y directions.	
		Interpretation		
		1010 (x direction) → We are uncertain about our x-position with a variance of 1010.
		1010 (y direction) → We are uncertain about our y-position with a variance of 1010.
		0 (off-diagonal) → There is no correlation between x and y uncertainty.

		Larger values in S mean we are more uncertain about our predicted position.
		As the filter runs, these values will decrease as we gain more confidence from measurements.   */
 					
        Matrix2d S = H * P * H.transpose() + R;
		
		/*  Compute the Kalman Gain (K)
		The Kalman Gain (K) determines how much we trust the sensor measurement versus the prediction.
		If sensor noise (R) is small, we trust the measurement more.
		If P is large (high uncertainty), we rely more on the measurement.
        Example Calculation:
									1000   0    0    0       1  0
		                     K  =	0    1000   0    0       0  1       1010   0    pow(-1) 
									0     0    1000  0    .  0	0   .    0    1010
									0     0     0   1000     0  0

                                   0.9901    0
                              K =     0    0.9901
                                      0      0
									  0      0
						This tells us we trust ~99% of the sensor reading for position, and velocity remains unchanged.	*/
						
        Matrix<double, 4, 2> K = P * H.transpose() * S.inverse(); // Kalman Gain

        /* Update state (x)
		This corrects the prediction based on how much we trust the measurement.
		Example Calculation:
								110    0.9901    0          
								 55        0   0.9901     -2   
							x =	 10  +     0     0     .   1
								  5        0     0
		
								110      -1.98
								 55       0.99
							x =	 10   +    0
								  5        0

								108.02
								 55.99
							x =	  10
								   5  
        Corrected position: (108.02, 55.99), velocity unchanged.		 */
        x = x + K * y;
		
		/*  Update the Covariance Matrix (P)
		This reduces uncertainty in the state estimate.
		The sensor measurement reduces our uncertainty about position, but not velocity.
		Effect:
		  P values for position will decrease.
          P values for velocity remain high (no direct measurement).  

		Matrix4d::Identity() creates a 4×4 identity matrix, which is a special square matrix where:
			All diagonal elements are 1.
			All off-diagonal elements are 0.		  */
		  
        P = (Matrix4d::Identity() - K * H) * P;
    }

    /* Extracts the first two elements of the state vector x, which represent position (x, y).
	head(2) means return only the first two elements (ignoring velocity).  
	The Kalman Filter tracks both position & velocity, 
	but sensors (e.g., cameras, LiDAR) often only measure position.*/
    Vector2d get_position() { return x.head(2); }
	
	/* Initializes only the position components (x, y) of x using the first measurement z (sensor reading).
       Velocity (vx, vy) is not set—it starts as zero or is estimated over time.
	   The Kalman Filter requires an initial state estimate.
       The first measurement (e.g., from a camera or GPS) gives an initial position.
       Velocity (vx, vy) might be unknown at startup, so it’s often set to zero initially.  */
    void initialize(const Vector2d& z) { x.head(2) = z; }

private:
    /* state vector, Represents the object's state (position and velocity).
			It is a 4×1 vector storing:
										𝑥
									x=	𝑦
										𝑣𝑥
										𝑣𝑦
			Example:
			x << 200, 300, 5, -2;
			The object is at (x=200, y=300).
			It moves at (vx=5, vy=-2).  */
    Vector4d x;

    /*  P — State Covariance Matrix,  
            Represents uncertainty in the state estimate
            It is a 4×4 matrix that tracks the variance of x, y, vx, vy and their correlations
            Higher values → Higher uncertainty			

        F — State Transition Matrix
            Defines how the object's state evolves over time
			It is initialized as an identity matrix and modified for constant velocity
									1  0  dt  0
									0  1   0  dt
								F =	0  0   1  0
									0  0   0  1
			How It Works?
			It updates position (x, y) based on velocity (vx, vy) and time step (dt).			
            Example Calculation (dt=0.1s):
             F(0, 2) = dt;
            Adds vx * dt to x and vy * dt to y. 
			
		Q - represents the process noise covariance, Q adds uncertainty to P in the prediction step
		    the uncertainty in the motion model due to unpredictable factors, 
			such as acceleration changes, sensor drift, or modeling inaccuracies	
			
			Without Q, the filter might become overconfident, 
			leading to errors when the motion model doesn't perfectly match real-world movements.
			Example of Q
			Q = Matrix4d::Identity() * 0.01;

			This assumes small process noise (0.01) for all state variables (x, y, vx, vy).
			It helps to gradually increase uncertainty over time.
			*/
			
    Matrix4d P, F, Q;
	
	/*	H — Measurement Matrix, defines how the sensor measures the object, extracts position (x, y) from state
		The sensor only observes position (x, y), not velocity:
				1 0 0 0
			H =	0 1 0 0
			Why?
			The sensor (e.g., a camera) doesn't measure velocity directly.
			H * x extracts only [x, y] from x. */
	
    Matrix<double, 2, 4> H;

    /* R — Measurement Covariance Matrix, represents sensor noise (uncertainty in measurements).
	It is a 2×2 matrix because the sensor measures only [x, y].
    Example Initialization
	  R = Matrix2d::Identity() * 10;
    Assumes sensor has a noise variance of 10 in both x and y. */
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

/* This function tracks objects detected by sensors using a Kalman Filter,
   groups them into clusters, and returns the estimated positions. 
		Input: A vector of sensor-detected objects (objects).
		Output: A vector of Cluster objects containing estimated positions.
   */
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
