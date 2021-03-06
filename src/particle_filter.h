/*
 * particle_filter.h
 *
 * 2D particle filter class.
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include "helper_functions.h"

struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};



class ParticleFilter {

	// Number of particles to draw
	int num_particles_;



	// Flag, if filter is initialized
	bool is_initialized_;

	// Vector of weights_ of all particles
	std::vector<double> weights_;

public:

	// Set of current particles
	std::vector<Particle> particles;

	// Constructor
	// @param num_particles_ Number of particles
	ParticleFilter() : num_particles_(0), is_initialized_(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * Init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 */
	void Init(double x, double y, double theta, double std[]);

	/**
	 * Prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], standard deviation of y [m]
	 *   standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void Prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);


	std::vector<LandmarkObs> GenerateValidLandmarks(Particle particle, const Map &map_landmarks, int sensor_range);
	std::vector<LandmarkObs> TransformVehicleToGlobal(Particle particle, const std::vector<LandmarkObs> &veh_objs);
	double ParticleWeight(LandmarkObs a, LandmarkObs b, double noise[]);
	void NormalizeParticleWeights();
	double DistanceBetweenLandmarks(LandmarkObs a, LandmarkObs b);
	void MatchObservationToClosestLandmark(Particle &particle, std::vector<LandmarkObs> observations,
										   std::vector<LandmarkObs> landmarks, double std_landmark[]);
	/**
	 * UpdateWeights Updates the weights for each particle based on the likelihood of the
	 *   observed measurements.
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void UpdateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations,
			const Map &map_landmarks);

	/**
	 * Resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void Resample();

	/*
	 * Set a particles list of associations, along with the associations calculated world x,y coordinates
	 * This can be a very useful debugging tool to make sure transformations are correct and assocations correctly connected
	 */
	Particle SetAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y);


	std::string GetAssociations(Particle best);
	std::string GetSenseX(Particle best);
	std::string GetSenseY(Particle best);

	/**
	* initialized Returns whether particle filter is initialized yet or not.
	*/
	const bool Initialized() const {
		return is_initialized_;
	}

	const int GetParticleCount() const {
		return num_particles_;
	}
};



#endif /* PARTICLE_FILTER_H_ */
