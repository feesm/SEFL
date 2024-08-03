#include <boost/qvm/lite.hpp>

constexpr boost::qvm::vec<float, 3> vnorm_acc = {0, 0, -1};
constexpr boost::qvm::vec<float, 3> vnorm_mag = {0.707F, 0.0F, 0.707F};

boost::qvm::vec<float, 3> f_gyro(boost::qvm::vec<float, 3> x, boost::qvm::vec<float, 3> u);
boost::qvm::mat<float, 3, 3> F_gyro(boost::qvm::vec<float, 3> x, boost::qvm::vec<float, 3> u);
boost::qvm::vec<float, 3> h_acc(boost::qvm::vec<float, 3> x);
boost::qvm::mat<float, 3, 3> H_acc(boost::qvm::vec<float, 3> x);
boost::qvm::vec<float, 3> h_mag(boost::qvm::vec<float, 3> x);
boost::qvm::mat<float, 3, 3> H_mag(boost::qvm::vec<float, 3> x);
