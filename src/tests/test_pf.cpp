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
    REQUIRE(pf.particles.size() == 100);

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
  SECTION("Predict position after movement - yaw rate = 0", "[Prediction]") {
    double dt = 0.1;
    double sig_pos[3] = {0.3, 0.3, 0.01};
    double v = 10;
    double ydd=0.0;

    std::vector<Particle> particles;
    for (int i=0; i<100; ++i) {
      Particle p;
      p.x = 0.0;
      p.y = 0.0;
      p.theta = 0.0;
      particles.push_back(p);
    }
    pf.particles = particles;
    pf.Prediction(dt, sig_pos, v, ydd);
    double sum_x = 0, sum_y = 0, sum_theta = 0;
    for (auto particle=pf.particles.begin(); particle != pf.particles.end(); ++particle) {
      sum_x += particle->x;
      sum_y += particle->y;
      sum_theta += particle->theta;
    }
    REQUIRE(sum_x/pf.particles.size() == Approx(1.0).margin(0.03)); 
    REQUIRE(sum_y/pf.particles.size() == Approx(0.0).margin(0.03)); 
    REQUIRE(sum_theta/pf.particles.size() == Approx(0.0).margin(0.001));

  }
}
