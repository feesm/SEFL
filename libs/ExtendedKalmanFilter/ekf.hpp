#include <boost/qvm/lite.hpp>

class EKF
{
    public:
        void predict(boost::qvm::vec<float, 3> u);
        void update(boost::qvm::vec<float, 3> z);

        boost::qvm::vec<float, 3> f(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u);
        boost::qvm::mat<float, 3, 3> F(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u);
        boost::qvm::vec<float, 3> h(const boost::qvm::vec<float, 3> &x);
        boost::qvm::mat<float, 3, 3> H(const boost::qvm::vec<float, 3> &x);

    private:
        boost::qvm::vec<float, 3> x;        //state estimate
        boost::qvm::mat<float, 3, 3> P;     //covariance estimate

        const boost::qvm::mat<float, 3, 3> Q;
        const boost::qvm::mat<float, 3, 3> R;

};
