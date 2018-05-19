//#define CATCH_CONFIG_MAIN
#include "./catch.hpp"
#include "../particle_filter.cpp"
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
