data {
 int<lower = 0> N; // number of observations
 real y[N]; // response variable
 real x[N]; // predictor variable
}

parameters {
  real a;
  real b;
  real<lower=0> sigma; // standard deviation
}

transformed parameters {
  real mu[N];
 
  for(i in 1:N)
    mu[i] <- a + b*x[i];
}

model {
 y ~ normal(mu, sigma);
}
