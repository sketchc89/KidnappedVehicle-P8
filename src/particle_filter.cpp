/*
 * particle_filter.cpp
 *
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

static std::default_random_engine rnd;

void ParticleFilter::Init(double x, double y, double theta, double std[]) {
  num_particles_ = 100;	

  //Add gaussian sensor noise to particle position
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

  //Generate gaussian sensor noise
  std::normal_distribution<double> x_dist(0, std_pos[0]);
  std::normal_distribution<double> y_dist(0, std_pos[1]);
  std::normal_distribution<double> t_dist(0, std_pos[2]);

  //Predict new position
  for (auto p=particles.begin(); p != particles.end(); ++p){
    if (fabs(yaw_rate) < 0.001) {
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

std::vector<LandmarkObs> ParticleFilter::GenerateValidLandmarks(Particle particle, const Map &map_landmarks, int sensor_range) {
  std::vector<LandmarkObs> valid_landmarks;
  double distance = 0;
    
  for (auto landmark=map_landmarks.landmark_list.begin(); landmark != map_landmarks.landmark_list.end(); ++landmark) {
    distance = std::sqrt(std::pow(landmark->x_f - particle.x, 2) +  std::pow(landmark->y_f - particle.y, 2));
    if (distance <= sensor_range) {
      LandmarkObs valid_landmark;
      valid_landmark.x = landmark->x_f;
      valid_landmark.y = landmark->y_f;
      valid_landmark.id = landmark->id_i;
      valid_landmarks.push_back(valid_landmark);
    }
  }
  return valid_landmarks;
}

std::vector<LandmarkObs> ParticleFilter::TransformVehicleToGlobal(Particle particle, const std::vector<LandmarkObs> &veh_objs) {
  
  std::vector<LandmarkObs> glob_objs;
  
  for (auto veh_obj=veh_objs.begin(); veh_obj != veh_objs.end(); ++veh_obj) {
    LandmarkObs glob_obj;
    glob_obj.x = veh_obj->x*std::cos(particle.theta) - 
      veh_obj->y*std::sin(particle.theta) +
      particle.x;
    glob_obj.y = veh_obj->x*std::sin(particle.theta) + 
      veh_obj->y*std::cos(particle.theta) +
      particle.y;
    glob_obj.id = veh_obj->id;
    glob_objs.push_back(glob_obj);
  }
  
  return glob_objs;
}

void ParticleFilter::NormalizeParticleWeights() {
  double total_weight = 0.0;

  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    total_weight += particle->weight;  
  }
  // if (total_weight < 0.01) {
  //     throw "Total weight of particles less than 0.01";
  // }
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    particle->weight /= total_weight;
  }
}

double ParticleFilter::ParticleWeight(LandmarkObs a, LandmarkObs b, double noise[]) {
  // Particle Weight is a gaussian
  return 1.0 / (2*M_PI*noise[0]*noise[1])*
             std::exp(-0.5*std::pow((a.x - b.x)/noise[0], 2) +
                      -0.5*std::pow((a.y - b.y)/noise[1], 2));
}

double ParticleFilter::DistanceBetweenLandmarks(LandmarkObs a, LandmarkObs b) {
  return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

void ParticleFilter::MatchObservationToClosestLandmark(Particle &particle, 
  std::vector<LandmarkObs> observations, std::vector<LandmarkObs> landmarks, double std_landmark[]) {
  
  bool init;
  double distance, min_dist;
  
  for (auto observation = observations.begin(); observation != observations.end(); ++observation) {
    min_dist = 0.0;
    init = false;
    for (auto landmark = landmarks.begin(); landmark != landmarks.end(); ++landmark) {
      distance = DistanceBetweenLandmarks(*landmark, *observation);
      if (!init) {
        init = true;
        min_dist = distance;
        particle.weight = ParticleWeight(*observation, *landmark, std_landmark);
      } else if (distance < min_dist) {
        min_dist = distance;
        particle.weight = ParticleWeight(*observation, *landmark, std_landmark);
      }
    }
  }
}

void ParticleFilter::UpdateWeights(double sensor_range, double std_landmark[], 
  const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  sensor_range = 100; // testing why landmarks missing
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
  }
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    std::vector<LandmarkObs> valid_landmarks = GenerateValidLandmarks(*particle, map_landmarks, sensor_range);
    std::vector<LandmarkObs> observed_landmarks = TransformVehicleToGlobal(*particle, observations);
    MatchObservationToClosestLandmark(*particle, observed_landmarks, valid_landmarks, std_landmark);
  }
  NormalizeParticleWeights();
}

void ParticleFilter::Resample() {

  std::vector<double> weights;  
  for (auto particle=particles.begin(); particle != particles.end(); ++particle) {
    weights.push_back(particle->weight);
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> w_dist(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;
  for (int i=0; i<num_particles_; ++i){
    Particle new_particle = particles[w_dist(gen)];
    new_particle.id = i;
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
