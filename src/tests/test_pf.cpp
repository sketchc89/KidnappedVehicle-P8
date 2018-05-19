//#define CATCH_CONFIG_MAIN
#include "./catch.hpp"
#include "../particle_filter.cpp"
const double M_EPS = 0.000001;
#ifndef M_PI
  #define M_PI 3.1415926535897;
#endif

TEST_CASE("Particle filter", "[particle_filter]") {
  ParticleFilter pf;
  Approx target = Approx(0.0).margin(0.3);

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
    REQUIRE(sum_x/pf.particles.size() == target); 
    REQUIRE(sum_y/pf.particles.size() == target); 
    REQUIRE(sum_theta/pf.particles.size() == target); 
  }
}
