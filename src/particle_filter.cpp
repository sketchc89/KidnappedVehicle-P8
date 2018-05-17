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
	std::normal_distribution<double> t_dist(z, std[2]);

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
	is_initialized_ = True;	
}

void ParticleFilter::Prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	double x, y, theta; //Predicted position
	
	//Predict new position
	for (auto p; p = particles.begin(); ++p){
		Particle new_particle;
		new_particle.x = std::cos(new_particle.theta)*velocity*delta_t;
		new_particle.y = std::sin(new_particle.theta)*velocity*delta_t;
		new_particle.theta = new_particle.theta*yaw_rate*delta_t;
	}
	
	//Add gaussian noise
	std::default_random_engine rnd;
	std::normal_distribution<double> x_dist(x, std_pos[0]);
	std::normal_distribution<double> y_dist(y, std_pos[1]);
	std::normal_distribution<double> t_dist(theta, std_pos[2]);
	
	//Generate particles
	for (int i=0; i<num_particles_; ++i){
		Particle cur_particle;
		cur_particle.id = i;
		cur_particle.x = x_dist(rnd);
		cur_particle.y = y_dist(rnd);
		cur_particle.theta = t_dist(rnd);
	
		particles.push_back(cur_particle);
	}

	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

}

void ParticleFilter::DataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the UpdateWeights phase.

}

void ParticleFilter::UpdateWeights_(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::Resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
}

string ParticleFilter::GetAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::GetSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::GetSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
