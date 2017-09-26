/*
 * particle_filter.cpp
 * Based on skeleton framework from Udacity self driving car 
 *    nanodegree program. Created by Tiffany Huang.
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
#include <map>

#include "particle_filter.h"
#include "map.h"

using namespace std;

bool debugging_enabled = false;

bool debug_iteration = false;
int iteration_num = 1;

// Some debug functions
void ParticleFilter::printParticles(string str) {
    
  if(debugging_enabled || (debug_iteration && iteration_num >= 905)){

    DEBUG(str);
      
    cout << "Number of particles:" << num_particles << endl;
    for (int i = 0; i < num_particles; ++i) {
      cout << "  Particle " << particles[i].id << " "
	   << "(" << particles[i].x << ","
	   << particles[i].y << ")"
	   << " theta:" << particles[i].theta
	   << " weight:" << particles[i].weight
	   << endl;
    }
  }

}

// Some debug functions
void ParticleFilter::printObservations(string str,
				       std::vector<LandmarkObs> observations){

  if(debugging_enabled || (debug_iteration && iteration_num >= 905)){

    DEBUG(str);

    cout << "Number of Observations:" << observations.size() << endl;
    for (int i = 0; i < observations.size(); i++){
      //Landmark observerd in vehicle coordinate system
      cout << "id " << observations[i].id << ": ("
	   << observations[i].x << ","
	   << observations[i].y << ")" << endl;
    }
  }

}

// Some debug functions
void ParticleFilter::printMapLandmarks(string str,
				       Map map_landmarks) {

  if(debugging_enabled || (debug_iteration && iteration_num >= 905)){

    DEBUG(str);
	
    std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
    
    cout << "Number of landmarks (in MAP coordinate system): "
	 << landmark_list.size() << endl;
    for (int i = 0; i < landmark_list.size(); ++i) {
      cout << "Landmark id: " << landmark_list[i].id_i;
      cout << "(" << landmark_list[i].x_f << ","
	   << landmark_list[i].y_f << ")" << endl;
    }
  }

}


void ParticleFilter::init(double gps_x, double gps_y,
			  double theta, double std[]) {
  // Set the number of particles.
  // Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.

  
  num_particles = 100; //Currently set to 100. 

  DEBUG("Init Values " << gps_x << " " << gps_y << " " << theta);
  
  default_random_engine gen;
  
  // Standard deviations for x, y, and psi
  double std_x, std_y, std_theta; 
  
  // Set standard deviations for x, y, and psi.
  std_x     = std[0];
  std_y     = std[1];
  std_theta = std[2];
  
  // Create a normal (Gaussian) distribution for x.
  normal_distribution<double> dist_x(gps_x, std_x);
  
  // Create a normal distributions for y
  normal_distribution<double> dist_y(gps_y, std_y);

  // Create a normal distributions for theta
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;
    
    // Sample from the normal distribution for each of x, y and theta
    sample_x     = dist_x(gen);
    sample_y     = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle particle;
    
    particle.id     = i;
    particle.x      = sample_x;
    particle.y      = sample_y;
    particle.theta  = sample_theta;
    particle.weight = 1;

    particles.push_back(particle);
    
  }

  //printMapLandmarks("** INIT: ", map_landmarks);
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
				double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // Add noise (along the same lines as during initialization above)

  DEBUG("****** PREDICTION: IN **********");
  // Set standard deviations for x, y, and psi.
  default_random_engine gen;
  double std_x, std_y, std_theta; // Standard deviations for x, y, and psi
  double x;
  double y;
  double theta;
  double ydt;
  
  // Set standard deviations for x, y, and psi.
  std_x     = std_pos[0];
  std_y     = std_pos[1];
  std_theta = std_pos[2];

  // Walk through all particles and predict
  for (int i = 0; i < num_particles; ++i) {
    
    x     = particles[i].x;
    y     = particles[i].y;
    theta = particles[i].theta;
    ydt   = yaw_rate * delta_t;

    //Handle small yaw_rate/division by zero
    if(fabs(yaw_rate) >= 1e-6) {

      x += (velocity/yaw_rate)*(sin(theta + ydt) - sin(theta));
      
      y += (velocity/yaw_rate)*(cos(theta) - cos(theta + ydt));

    } else {

      x += velocity * delta_t * cos(theta);
      
      y += velocity * delta_t * sin(theta);
      
    }
    
    theta = theta + ydt;
      
    // Create a normal (Gaussian) distribution for x.
    normal_distribution<double> dist_x(x, std_x);
    
    // Create a normal distributions for y
    normal_distribution<double> dist_y(y, std_y);
    
    // Create a normal distributions for theta
    normal_distribution<double> dist_theta(theta, std_theta);
    
    particles[i].x     = dist_x(gen);
    particles[i].y     = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    
  }

  printParticles("** PREDICTION: ");

  DEBUG("****** PREDICTION: OUT **********");
  
}

// 1. Observations in vehicle coordinate system
// 2. Particle location in map coordinte system
// 3. Rotate based on the orientation of the particle
// 4. Translate to the location of the particle 
// Equations:
// For each particle at (xl, yl) and with theta:
//       For each observation at (xo, yo) in vehicle coordinate system
//          xm = xo*cos(theta) - yo*sin(theta) + xl
//          ym = xo*sin(theta) + yo*cos(theta) + yl
void ParticleFilter::transformObsToMap(Particle particle,
				       std::vector<LandmarkObs> &observations){
  double xl, xo, xm;
  double yl, yo, ym;
  double theta, sin_theta, cos_theta;
    
  xl    = particle.x;
  yl    = particle.y;
  theta = particle.theta;
  
  sin_theta = sin(theta);
  cos_theta = cos(theta);
  
  //For each observation
  for (int i = 0; i < observations.size(); i++){
    //Transform to map coordinate system
    xo = observations[i].x;
    yo = observations[i].y;
    
    xm = xo*cos_theta - yo*sin_theta + xl;
    ym = xo*sin_theta + yo*cos_theta + yl;
    
    //Overload observations.x and y - set to map coordinates  
    observations[i].x = xm;
    observations[i].y = ym;
    
    DEBUG("Obs id: " << observations[i].id);
    
    DEBUG("  Vehicle:  (" << xo << "," << yo << ")");
    
    DEBUG("  Map    :  (" << xm << "," << ym << ")");
    
  }
  
}

double ParticleFilter::computeWeight_old(std::vector<LandmarkObs> observations,
					 std::vector<Map::single_landmark_s> associated_landmarks,
				     double std_landmarks[]){
  double prod = 1;
  double stdx = std_landmarks[0];
  //Note error in description of updateWeights() in particle_filter.h
  // Second element of the std_landmarks is y measurement uncertainty
  double stdy = std_landmarks[1]; 
  double x, y, xmu, ymu;
  double power, powerx, powery;
  DEBUG("computeWeight: IN");
  
  for (int i = 0; i < observations.size(); ++i) {
    x = observations[i].x;
    y = observations[i].y;

    xmu = associated_landmarks[i].x_f;
    ymu = associated_landmarks[i].y_f;
    
    powerx = (x-xmu)*(x-xmu)/(2*(stdx*stdx));
    powery = (y-ymu)*(y-ymu)/(2*(stdy*stdy));
    power = -(powerx + powery);
    prod *= (1/(2*M_PI*stdx*stdy))*exp(power);
  }
  
  DEBUG("computeWeight: OUT");

  return prod;
}

double ParticleFilter::computeWeight(Particle particle,
				     std::vector<LandmarkObs> observations,
				     double std_landmarks[]){
  double prod = 1;
  double stdx = std_landmarks[0];
  //Note error in description of updateWeights() in particle_filter.h
  // Second element of the std_landmarks is y measurement uncertainty
  double stdy = std_landmarks[1]; 
  double x, y, xmu, ymu;
  double power, powerx, powery;
  DEBUG("computeWeight: IN");
  
  for (int i = 0; i < observations.size(); ++i) {
    
    x = observations[i].x;
    y = observations[i].y;

    xmu = particle.sense_x[i];
    ymu = particle.sense_y[i];
    
    powerx = (x-xmu)*(x-xmu)/(2*(stdx*stdx));
    powery = (y-ymu)*(y-ymu)/(2*(stdy*stdy));
    power = -(powerx + powery);
    prod *= (1/(2*M_PI*stdx*stdy))*exp(power);
  }
  
  DEBUG("computeWeight: OUT");

  return prod;
}

void ParticleFilter::dataAssociation_old(std::vector<LandmarkObs>& observations,
					 std::vector<Map::single_landmark_s> &associated_landmarks,
				     Map map_landmarks) {
  // Assign the  observed measurement to the landmark that is closes to it

  double xm, ym;
  std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
  
  Map::single_landmark_s best_landmark = {};
  double best_landmark_dist = 0;

  //Walk through all observations
  for (int i = 0; i < observations.size(); ++i) {
    xm = observations[i].x;
    ym = observations[i].y;

    DEBUG("Association of observations: " << observations[i].id);
    DEBUG("  Map    :  (" << xm << "," << ym << ")");
    
    //Initialize
    best_landmark.id_i = landmark_list[0].id_i;
    best_landmark.x_f  = landmark_list[0].x_f;
    best_landmark.y_f  = landmark_list[0].y_f;

    //Initialize the eucledian distance
    best_landmark_dist = dist(xm, ym, landmark_list[0].x_f, landmark_list[0].y_f);

    //Find which landmark is closes to this observation
    for (int j = 0; j < landmark_list.size(); ++j) {
      //Find the eucladian distance
      double this_dist = dist (xm, ym, landmark_list[j].x_f, landmark_list[j].y_f);

      if(this_dist < best_landmark_dist){
	best_landmark.id_i = landmark_list[j].id_i;
	best_landmark.x_f  = landmark_list[j].x_f;
	best_landmark.y_f  = landmark_list[j].y_f;
	
	best_landmark_dist = this_dist;
      }
    }

    associated_landmarks[i].id_i = best_landmark.id_i;
    associated_landmarks[i].x_f  = best_landmark.x_f;
    associated_landmarks[i].y_f  = best_landmark.y_f;
    
    DEBUG("  Assoc  :  (" << associated_landmarks[i].x_f
	  << "," << associated_landmarks[i].y_f
	  << ")");
    
  }  

}

void ParticleFilter::dataAssociation(Particle &particle,
				     std::vector<LandmarkObs>& observations,
				     Map map_landmarks) {

  // Assign the  observed measurement to the landmark that is closes to it

  double xm, ym;
  std::vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;

  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  int best_landmark_id      = 0;
  double best_landmark_dist = 0;

  //Walk through all observations
  for (int i = 0; i < observations.size(); ++i) {
    xm = observations[i].x;
    ym = observations[i].y;

    DEBUG("Association of observations: " << observations[i].id);
    DEBUG("  Map    :  (" << xm << "," << ym << ")");

    //Init
    best_landmark_id = 0;
    //Initialize the eucledian distance
    best_landmark_dist = dist(xm, ym, landmark_list[0].x_f, landmark_list[0].y_f);

    //Find which landmark is closes to this observation
    for (int j = 0; j < landmark_list.size(); ++j) {
      //Find the eucladian distance
      double this_dist = dist (xm, ym, landmark_list[j].x_f, landmark_list[j].y_f);

      if(this_dist < best_landmark_dist){
	best_landmark_id = j;
	best_landmark_dist = this_dist;
      }
    }

    DEBUG("  Assoc  :  (" << landmark_list[best_landmark_id].x_f
	  << "," << landmark_list[best_landmark_id].y_f
	  << ")");
    particle.associations.push_back(landmark_list[best_landmark_id].id_i);
    particle.sense_x.push_back(landmark_list[best_landmark_id].x_f);
    particle.sense_y.push_back(landmark_list[best_landmark_id].y_f);
    
  }

    

}


//Copy observtions from dst to src
void ParticleFilter::copyObservation(std::vector<LandmarkObs> &dst,
				     std::vector<LandmarkObs> src){
 
  for (int i = 0; i < src.size(); ++i) {
    LandmarkObs o;

    o.x  = src[i].x;
    o.y  = src[i].y;
    o.id = src[i].id;
    dst.push_back(o);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
				   std::vector<LandmarkObs> observations,
				   Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian
  // distribution.
  // Ref::
  //       https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE:
  //   The observations are in the VEHICLE'S coordinate system.
  //    Particles are located in the MAP'S coordinate system.
  //    Transofmraiton required.
  //   Ref: https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   http://planning.cs.uiuc.edu/node99.html

  std::vector<Map::single_landmark_s> associated_landmarks(observations.size());

  cout << "Iteration:  " << iteration_num++ << endl;


  printObservations("Observations (in VEHICLE coordinate system):",
		    observations);


  weights.clear();
  
  // Walk through all particles
  for (int i = 0; i < num_particles; ++i) {

    std::vector<LandmarkObs> obs;
    
    //Copy observations into obs for this particle
    copyObservation(obs, observations);

    DEBUG("****** Update Weight Particle: " << particles[i].id << " **********");
    
    //Transform the observations to MAP coordinate system w.r.t Particle
    transformObsToMap(particles[i], obs);

    //NOTE: Re-did the functions based on associations list in the particle.
    //Make associations: Which landmark is this observation closest to?
    dataAssociation(particles[i], obs, map_landmarks);
    
    //Compute probability. Note: Redid this function.
    particles[i].weight = computeWeight(particles[i], obs, std_landmark);
    
    //NOTE: Interesting that resample (discrete_distribution) works with
    // push_back. Does not seem to work when the weights are reserve() upfront
    // TODO: Investigate!!
    
    weights.push_back(particles[i].weight);

    DEBUG("****** Update Weight Particle: DONE " << particles[i].id << "********");
    
  }

  printParticles("** FINAL PARTICLE WEIGHTS:");
  

}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their
  //   weight. 
  // NOTE: Reference: std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  DEBUG("IN RESAMPLE");

  vector<Particle> resampled_particles;

  default_random_engine gen;

  discrete_distribution<int> distribution(weights.begin(), weights.end()); 
  
  for (int i = 0; i < particles.size(); i++) {
    int index = distribution(gen);
    particles[index].id = i;
    resampled_particles.push_back(particles[index]);
    
  }

  particles = resampled_particles;
  
  printParticles("*** SAMPLED PARTICLES");
  
  //TODO: Anything needs to be done to free memory? Investigate!!
  
  //throw std::exception();  
}

Particle ParticleFilter::SetAssociations(Particle particle,
					 std::vector<int> associations,
					 std::vector<double> sense_x,
					 std::vector<double> sense_y)
{
  //Particle: the particle to assign each listed association, and
  //  association's (x,y) world coordinates mapping to
  //associations: The landmark id that goes along with each listed association
  //sense_x: the associations x mapping already converted to world coordinates
  //sense_y: the associations y mapping already converted to world coordinates
  
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
  
  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
