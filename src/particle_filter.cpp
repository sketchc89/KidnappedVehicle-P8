/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

//using namespace std;

void ParticleFilter::Init(double x, double y, double theta, double std[]) {
	num_particles_ = 100;	
	
	//Add gaussian noise
	std::default_random_engine rnd;
	std::normal_distribution<double> x_dist(x, std[0]);
	std::normal_distribution<double> y_dist(y, std[1]);
	std::normal_distribution<double> t_dist(theta, std[2]);

	//Generate particles
	for (int i=0; i<num_particles_; ++i){
		Particle cur_particle;
		cur_particle.id = i;
		cur_particle.x = x_dist(rnd);
		cur_particle.y = y_dist(rnd);
		cur_particle.theta = t_dist(rnd);
		cur_particle.weight = 1.;

		particles.push_back(cur_particle);
	}
	
	//Initialization complete
	is_initialized_ = true;	
}
void ParticleFilter::Prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	//Generate gaussian noise
	std::default_random_engine rnd;
	std::normal_distribution<double> x_dist(0, std_pos[0]);
	std::normal_distribution<double> y_dist(0, std_pos[1]);
	std::normal_distribution<double> t_dist(0, std_pos[2]);
	
	//Predict new position
	for (auto p=particles.begin(); p != particles.end(); ++p){
		if (yaw_rate < 0.001) {
			p->x += std::cos(p->theta)*velocity*delta_t + x_dist(rnd);
			p->y += std::sin(p->theta)*velocity*delta_t + y_dist(rnd);
			p->theta += t_dist(rnd);
		} else {
			p->x += velocity/yaw_rate*(std::sin(p->theta + yaw_rate*delta_t) - std::sin(p->theta)) + x_dist(rnd);
			p->y += velocity/yaw_rate*(std::cos(p->theta) - std::cos(p->theta + yaw_rate*delta_t)) + y_dist(rnd);
			p->theta += yaw_rate*delta_t + t_dist(rnd);
		}
	}
}

void ParticleFilter::DataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
	double distance;
	double min_dist;
	bool init;
	for (auto feature=observations.begin(); feature != observations.end(); ++feature) {
		init=false;
		for (auto prediction=predicted.begin(); prediction != predicted.end(); ++prediction) {
			distance = std::sqrt(std::pow(prediction->x - feature->x, 2) + 
								std::pow(prediction->y - feature->y, 2));
			if (!init || distance < min_dist){
				init = true;
				min_dist = distance;
				feature->id = prediction->id;
			}
		}
	}
}

void ParticleFilter::UpdateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	double distance;
  double total_weight=0.0;
	for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
		//Iterate through landmarks and build landmark vector of landmarks within sensor range
		std::vector<LandmarkObs> map_predictions;
		for (auto landmark=map_landmarks.landmark_list.begin(); landmark != map_landmarks.landmark_list.end(); ++landmark) {
			distance = std::sqrt(std::pow(landmark->x_f - particle->x, 2) +
								std::pow(landmark->y_f - particle->y, 2));
			if (distance <= sensor_range) {
				LandmarkObs valid_landmark;
				valid_landmark.x = landmark->x_f;
				valid_landmark.y = landmark->y_f;
				valid_landmark.id = landmark->id_i;
				map_predictions.push_back(valid_landmark);
			}
		}

		// Transform all observations in car reference frame to map reference frame
		std::vector<LandmarkObs> map_observations;
		for (auto observation=observations.begin(); observation != observations.end(); ++observation) {
			LandmarkObs map_observation;
			map_observation.x = observation->x*std::cos(particle->theta) - 
								          observation->y*std::sin(particle->theta) +
								          particle->x;
			map_observation.y = observation->x*std::sin(particle->theta) + 
								          observation->y*std::cos(particle->theta) +
								          particle->y;
			map_observation.id = observation->id;
			map_observations.push_back(map_observation);
		}

		//Associate predicted position of landmark with closest observed position
		DataAssociation(map_predictions, map_observations);
    for (auto observation=map_observations.begin(); observation != map_observations.end(); ++observation) {
      for (auto prediction=map_predictions.begin(); prediction != map_predictions.end(); ++prediction) {
        if (observation->id == prediction->id) {
          particle->weight = 1.0 / (2*M_PI*std_landmark[0]*std_landmark[1])*
                             std::exp(-0.5*std::pow((prediction->x - observation->x)/std_landmark[0], 2)+
                                           std::pow((prediction->y - observation->y)/std_landmark[1], 2));
        }
      }
    }
    total_weight += particle->weight;
	}
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    particle->weight /= total_weight;
  }
}

void ParticleFilter::Resample() {
  std::default_random_engine rnd;
  std::vector<double> weights;
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    weights.push_back(particle->weight);
  }
  std::discrete_distribution<> w_dist(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles;
	for (int i=0; i<num_particles_; ++i){
		Particle new_particle = particles[w_dist(rnd)];
		resampled_particles.push_back(new_particle);
	}
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

std::string ParticleFilter::GetAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::GetSenseX(Particle best)
{
  std::vector<double> v = best.sense_x;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::GetSenseY(Particle best)
{
  std::vector<double> v = best.sense_y;
	std::stringstream ss;
    std::copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
