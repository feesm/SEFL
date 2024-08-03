#include "ekf.hpp"

using namespace boost::qvm;

void EKF::predict(boost::qvm::vec<float, 3> u)
{
    auto F_k= F(x,u);
    P =F_k * P * transposed(F_k) + Q;               //Predicted covariance estimate
    x += f(x,u);                                        //Predicted state estimate
}

void EKF::update(boost::qvm::vec<float, 3> z)
{
    auto y = z - h(x);                                  //measurement residual
    auto H_k = H(x);      
    auto S = H_k * P * transposed(H_k) + R;         //residual covariance
    auto K = P * transposed(H_k) * inverse(S);   //Kalman gain

    x = x + K * y;                                      //Updated state estimate
    P = (identity_mat<float, 3>() - K * H_k) * P;       //Updated covariance estimate
}

vec<float, 3> EKF::f(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u)
{
    float s_x0 = sin(A<0> (x));
    float c_x0 = cos(A<0> (x));
    float s_x1 = sin(A<1> (x));
    float c_x1 = cos(A<1> (x));
    float t_x1 = s_x1 / c_x1;
   
    vec<float, 3> f = zero_vec<float, 3>();

    A<0>(f) = A<0>(u) + (A<1>(u) * s_x0 + A<2>(u) * c_x0) * t_x1;
    A<1>(f) = A<1>(u) * c_x0 - A<2>(u) * s_x0;
    A<2>(f) = (A<1>(u) * s_x0 + A<2>(u) * c_x0) / c_x1;
    return f;
}

mat<float, 3, 3> EKF::F(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u)
{
    mat<float, 3, 3> jacobian = zero_mat<float, 3>();
    
    float s_x0 = sin(A<0> (x));
    float c_x0 = cos(A<0> (x));
    float s_x1 = sin(A<1> (x));
    float c_x1 = cos(A<1> (x));
    float t_x1 = s_x1 / c_x1;
    float sec_x1 = 1.0F / c_x1;

    A<0, 0>(jacobian) = (A<1>(u) * c_x0 - A<2>(u) * s_x0) * t_x1;
    A<0, 1>(jacobian) = (A<1>(u) * s_x0 + A<2>(u) * c_x0) * (t_x1 * t_x1 + 1.0F);
    A<1, 0>(jacobian) = - A<1>(u) * s_x0 - A<2>(u) * c_x0;
    A<2, 0>(jacobian) = (A<1>(u) * c_x0 - A<2>(u) * s_x0) * sec_x1;
    A<2, 1>(jacobian) = (A<1>(u) * s_x0 + A<2>(u) * c_x0) * t_x1 * sec_x1;
   
    return jacobian;
}

vec<float, 3> EKF::h(const boost::qvm::vec<float, 3> &x)
{
    float s_x0 = sin(A<0>(x));
    float c_x0 = cos(A<0>(x));
    float s_x1 = sin(A<1>(x));
    float c_x1 = cos(A<1>(x));
    float s_x2 = sin(A<2>(x));
    float c_x2 = cos(A<2>(x));

    mat<float, 3, 3> R_x = identity_mat<float, 3>();
    A<1, 1>(R_x) = c_x0;
    A<1, 2>(R_x) = s_x0;
    A<2, 1>(R_x) = -s_x0;
    A<2, 2>(R_x) = c_x0;

    mat<float, 3, 3> R_y = identity_mat<float, 3>();
    A<0, 0>(R_y) = c_x1;
    A<0, 2>(R_y) = -s_x1;
    A<2, 0>(R_y) = s_x1;
    A<2, 2>(R_y) = c_x1;

    mat<float, 3, 3> R_z = identity_mat<float, 3>();
    A<0, 0>(R_z) = c_x2;
    A<0, 1>(R_z) = s_x2;
    A<1, 0>(R_z) = -s_x2;
    A<1, 1>(R_z) = c_x2;

    vec<float, 3> v = zero_vec<float, 3>();
    A<2>(v) = -1;
    
    vec<float, 3> h = R_x * R_y * R_z * v;
    return h;
}

mat<float, 3, 3> EKF::H(const boost::qvm::vec<float, 3> &x)
{
    float s_x0 = sin(A<0>(x));
    float c_x0 = cos(A<0>(x));
    float s_x1 = sin(A<1>(x));
    float c_x1 = cos(A<1>(x));
    float s_x2 = sin(A<2>(x));
    float c_x2 = cos(A<2>(x));

    mat<float, 3, 3> R_x = identity_mat<float, 3>();
    A<1, 1>(R_x) = c_x0;
    A<1, 2>(R_x) = s_x0;
    A<2, 1>(R_x) = -s_x0;
    A<2, 2>(R_x) = c_x0;

    mat<float, 3, 3> R_y = identity_mat<float, 3>();
    A<0, 0>(R_y) = c_x1;
    A<0, 2>(R_y) = -s_x1;
    A<2, 0>(R_y) = s_x1;
    A<2, 2>(R_y) = c_x1;

    mat<float, 3, 3> R_z = identity_mat<float, 3>();
    A<0, 0>(R_z) = c_x2;
    A<0, 1>(R_z) = s_x2;
    A<1, 0>(R_z) = -s_x2;
    A<1, 1>(R_z) = c_x2;

    mat<float, 3, 3> dR_x = identity_mat<float, 3>();
    A<1, 1>(dR_x) = -s_x0;
    A<1, 2>(dR_x) = c_x0;
    A<2, 1>(dR_x) = -c_x0;
    A<2, 2>(dR_x) = -s_x0;

    mat<float, 3, 3> dR_y = identity_mat<float, 3>();
    A<0, 0>(dR_y) = -s_x1;
    A<0, 2>(dR_y) = -c_x1;
    A<2, 0>(dR_y) = c_x1;
    A<2, 2>(dR_y) = -s_x1;

    mat<float, 3, 3> dR_z = identity_mat<float, 3>();
    A<0, 0>(dR_z) = -s_x2;
    A<0, 1>(dR_z) = c_x2;
    A<1, 0>(dR_z) = -c_x2;
    A<1, 1>(dR_z) = -s_x2;

    vec<float, 3> v = zero_vec<float, 3>();
    A<2>(v) = -1;

    mat<float, 3, 3> jacobian = zero_mat<float, 3>();
    col<0>(jacobian) = dR_x * R_y * R_z * v;
    col<1>(jacobian) = R_x * dR_y * R_z * v;
    col<2>(jacobian) = R_x * R_y * dR_z * v;

    return jacobian;
}
