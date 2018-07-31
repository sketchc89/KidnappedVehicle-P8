//#define CATCH_CONFIG_MAIN
#include "./catch.hpp"
#include "../particle_filter.cpp"
#include "../helper_functions.h"
const double M_EPS = 0.000001;
#ifndef M_PI
  #define M_PI 3.1415926535897;
#endif

TEST_CASE("Particle filter", "[particle_filter]") {
  ParticleFilter pf;

  double x=0, y=0, theta=0;
  double variance[3] = {0.3, 0.3, 0.01};

  SECTION("Initialize particle filter") {
    pf.Init(x, y, theta, variance);
    REQUIRE(pf.particles.size() == pf.GetParticleCount());

    double sum_x = 0, sum_y = 0, sum_theta = 0;
    for (auto particle=pf.particles.begin(); particle != pf.particles.end(); ++particle) {
      sum_x += particle->x;
      sum_y += particle->y;
      sum_theta += particle->theta;
    }
    Approx target = Approx(0.0).margin(0.3);
    REQUIRE(sum_x/pf.particles.size() == target); 
    REQUIRE(sum_y/pf.particles.size() == target); 
    REQUIRE(sum_theta/pf.particles.size() == target);
  }
  
  double no_variance[3] = {0.0, 0.0, 0.0};
  pf.Init(x, y, theta, no_variance);

  SECTION("Predict position after movement - yaw rate = 0", "[Prediction]") {
    double dt = 0.1;
    double sig_pos[3] = {0.3, 0.3, 0.01};
    double v = 10;
    double ydd=0.0;

    pf.Prediction(dt, sig_pos, v, ydd);
    double sum_x = 0, sum_y = 0, sum_theta = 0;
    for (auto particle=pf.particles.begin(); particle != pf.particles.end(); ++particle) {
      sum_x += particle->x;
      sum_y += particle->y;
      sum_theta += particle->theta;
    }
    // std::cout << "Sum x, y, theta\t" << sum_x << "\t" << sum_y << "\t" 
    //           << sum_theta << "\t" << pf.particles.size() << "\n";
    REQUIRE(sum_x/pf.particles.size() == Approx(1.0).margin(0.03)); 
    REQUIRE(sum_y/pf.particles.size() == Approx(0.0).margin(0.03)); 
    REQUIRE(sum_theta/pf.particles.size() == Approx(0.0).margin(0.001));

  }

  SECTION("Predict position after movement - yaw rate, aggressive", "[Prediction]") {
    double dt = 1;
    double sig_pos[3] = {0.3, 0.3, 0.01};
    double v = 10;
    double ydd=0.35; //20 degrees

    pf.Prediction(dt, sig_pos, v, ydd);
    double sum_x = 0, sum_y = 0, sum_theta = 0;
    for (auto particle=pf.particles.begin(); particle != pf.particles.end(); ++particle) {
      sum_x += particle->x;
      sum_y += particle->y;
      sum_theta += particle->theta;
    }
    // std::cout << "Sum x, y, theta\t" << sum_x << "\t" << sum_y << "\t" 
    //           << sum_theta << "\t" << pf.particles.size() << "\n";
    REQUIRE(sum_x/pf.particles.size() == Approx(9.797).margin(0.03)); 
    REQUIRE(sum_y/pf.particles.size() == Approx(1.732).margin(0.03)); 
    REQUIRE(sum_theta/pf.particles.size() == Approx(0.35).margin(0.001));

  }

  
  SECTION("Associates data when prediction and observation match", "[DataAssociation]") {
    LandmarkObs obs_0 = {-1, 0.0, 0.0}, pred_0 = {0, 0.0, 0.0}, 
                obs_1 = {-1, 1.0, 0.0}, pred_1 = {1, 1.0, 0.0},
                obs_2 = {-1, 0.0, 1.0}, pred_2 = {2, 0.0, 1.0},
                obs_3 = {-1, 1.0, 1.0}, pred_3 = {3, 1.0, 1.0},
                obs_4 = {-1, -1.0, 0.0}, pred_4 = {4, -1.0, 0.0},
                obs_5 = {-1, 0.0, -1.0}, pred_5 = {5, 0.0, -1.0},
                obs_6 = {-1, -1.0, -1.0}, pred_6 = {6, -1.0, -1.0};
    
    
    std::vector<LandmarkObs> observations = {obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6};
    std::vector<LandmarkObs> predictions = {pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6};
    pf.DataAssociation(predictions, observations);
    for (int i = 0; i < observations.size(); ++i) {
      REQUIRE(observations[i].id == i);
    }
  }
  
  SECTION("Associates observations with the correct prediction", "[DataAssociation]") {
    LandmarkObs obs_0 = {-1, 0.3, 0.1}, pred_0 = {0, 0.0, 0.0}, 
                obs_1 = {-1, 1.3, 0.1}, pred_1 = {1, 1.0, 0.0},
                obs_2 = {-1, 0.3, 1.0}, pred_2 = {2, 0.0, 1.0},
                obs_3 = {-1, 1.3, 1.3}, pred_3 = {3, 1.0, 1.0},
                obs_4 = {-1, -1.3, 0.1}, pred_4 = {4, -1.0, 0.0},
                obs_5 = {-1, 0.1, -1.3}, pred_5 = {5, 0.0, -1.0},
                obs_6 = {-1, -1.3, -1.3}, pred_6 = {6, -1.0, -1.0};
    
    std::vector<LandmarkObs> observations = {obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6};
    std::vector<LandmarkObs> predictions = {pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6};
    pf.DataAssociation(predictions, observations);
    for (int i = 0; i < observations.size(); ++i) {
      REQUIRE(observations[i].id == i);
    }
  }

  /* Determined root cause of issue is in UpdateWeights
  SECTION("Update weights when observations and predictions match exactly and no noise", "[UpdateWeights]"){
    LandmarkObs obs_0 = {-1, 0.0, 0.0}, 
                obs_1 = {-1, 1.0, 0.0}, 
                obs_2 = {-1, 0.0, 1.0}, 
                obs_3 = {-1, 1.0, 1.0}, 
                obs_4 = {-1, -1.0, 0.0}, 
                obs_5 = {-1, 0.0, -1.0}, 
                obs_6 = {-1, -1.0, -1.0}; 
    
    double sensor_range = 10.0;
    double sigma_landmark[2] = {0.0, 0.0};
    std::vector<LandmarkObs> observations = {obs_0, obs_1, obs_2, obs_3, obs_4, obs_5, obs_6};
    Map predictions;
    read_map_data("./map_data.txt", predictions);
    for (int i=0; i<pf.GetParticleCount(); ++i) {
      std::cout << "Particle\t" << i << "\tWeight:\t" << pf.particles[i].weight << "\n";
    }
    pf.UpdateWeights(sensor_range, sigma_landmark, observations, predictions);
    
    for (int i=0; i<pf.GetParticleCount(); ++i) {
      std::cout << "Particle\t" << i << "\tWeight:\t" << pf.particles[i].weight << "\n";
    }
  }*/

  SECTION("Resample particles", "[Resample]") {
    int ratio = 10;
    std::vector<Particle> resampled_particles;
    for (int i=0; i<pf.GetParticleCount(); ++i) {
      Particle p;
      if (i % ratio == 0) {
        p.x = 10.0;
        p.y = 10.0;
        p.theta = 1.0;
      } else {
        p.x = 0.0;
        p.y = 0.0;
        p.theta = 0.0;
      }
      p.weight = 1;
      resampled_particles.push_back(p);
    }
    pf.particles = resampled_particles;
  
    std::vector<Particle> old_particles = pf.particles;
    pf.Resample();
    REQUIRE(pf.particles.size() == pf.GetParticleCount());
    
    int count = 0;
    for (auto p = pf.particles.begin(); p != pf.particles.end(); ++p) {
      if (p->x == 10 && p->y == 10 && p->theta == 1.0) {
        count++;
      }
    }
    // std::cout << "\nCount:\t"<< count << "\n";
    REQUIRE(count < 1.5*(pf.GetParticleCount()/ratio));
  }
}
