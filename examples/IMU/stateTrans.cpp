#include "stateTrans.hpp"

using namespace boost::qvm;

vec<float, 3> f_gyro(const boost::qvm::vec<float, 3> x, const boost::qvm::vec<float, 3> u)
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

mat<float, 3, 3> F_gyro(const boost::qvm::vec<float, 3> x, const boost::qvm::vec<float, 3> u)
{
    mat<float, 3, 3> jacobian = zero_mat<float, 3>();
    
    float s_x0 = sin(A<0> (x));
    float c_x0 = cos(A <0> (x));
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

vec<float, 3> h_acc(const boost::qvm::vec<float, 3> x)
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

    vec<float, 3> h = R_x * R_y * R_z * vnorm_acc;
    return h;
}

mat<float, 3, 3> H_acc(const boost::qvm::vec<float, 3> x)
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
    col<0>(jacobian) = dR_x * R_y * R_z * vnorm_acc;
    col<1>(jacobian) = R_x * dR_y * R_z * vnorm_acc;
    col<2>(jacobian) = R_x * R_y * dR_z * vnorm_acc;

    return jacobian;
}
vec<float, 3> h_mag(const boost::qvm::vec<float, 3> x)
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

    vec<float, 3> h = R_x * R_y * R_z * vnorm_mag;
    return h;
}

mat<float, 3, 3> H_mag(const boost::qvm::vec<float, 3> x)
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
    col<0>(jacobian) = dR_x * R_y * R_z * vnorm_mag;
    col<1>(jacobian) = R_x * dR_y * R_z * vnorm_mag;
    col<2>(jacobian) = R_x * R_y * dR_z * vnorm_mag;

    return jacobian;
}
