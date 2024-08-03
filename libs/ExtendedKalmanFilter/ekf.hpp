#include <boost/qvm/lite.hpp>


namespace sefl
{
    template<class T, int n>
    class ekf
    {
        public:
            ekf(boost::qvm::mat<T, n, n> Q_0, boost::qvm::mat<T, n, n> R_0,
                    boost::qvm::mat<T, n, n> P_0 = boost::qvm::identity_mat<T, n>(),
                    boost::qvm::vec<T, n> x_0 = boost::qvm::zero_vec<T, n>()):
                x(x_0), P(P_0), Q(Q_0), R(R_0) {}


            template<int m> 
            void predict(boost::qvm::vec<T, m> u, T dt,
                    boost::qvm::vec<T, n> (*f)(boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u),
                    boost::qvm::mat<T, n, n> (*F) (boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u));

            template<int m>
            void update(boost::qvm::vec<T, m> z,
                    boost::qvm::vec<T, m> (*h) (boost::qvm::vec<T, n> x),
                    boost::qvm::mat<T, m, m> (*H) (boost::qvm::vec<T, n> x));

            boost::qvm::vec<float, 3> f(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u);
            boost::qvm::mat<float, 3, 3> F(const boost::qvm::vec<float, 3> &x, const boost::qvm::vec<float, 3> &u);
            boost::qvm::vec<float, 3> h(const boost::qvm::vec<float, 3> &x);
            boost::qvm::mat<float, 3, 3> H(const boost::qvm::vec<float, 3> &x);

        private:
            boost::qvm::vec<T, n> x;        //state estimate
            boost::qvm::mat<T, n, n> P;     //covariance estimate

            const boost::qvm::mat<T, n, n> Q;
            const boost::qvm::mat<T, n, n> R;
    };
}

template<class T, int n>
template<int m>
void sefl::ekf<T, n>::predict(boost::qvm::vec<T, m> u, T dt,
                    boost::qvm::vec<T, n> (*f)(boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u),
                    boost::qvm::mat<T, n, n> (*F) (boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u))
{
    boost::qvm::mat<T, n, n> F_k= F(x,u);
    P += (F_k * P + P * transposed(F_k) + Q) * dt;              //Predicted covariance estimate
    x += f(x,u) * dt;                                           //Predicted state estimate
}

template<class T, int n>
template<int m>
void sefl::ekf<T, n>::update(boost::qvm::vec<T, m> z,
                    boost::qvm::vec<T, m> (*h) (boost::qvm::vec<T, n> x),
                    boost::qvm::mat<T, m, m> (*H) (boost::qvm::vec<T, n> x))
{
    auto y = z - h(x);                                          //measurement residual
    auto H_k = H(x);      
    auto S = H_k * P * transposed(H_k) + R;                     //residual covariance
    auto K = P * transposed(H_k) * inverse(S);                  //Kalman gain

    x = x + K * y;                                              //Updated state estimate
    P = (boost::qvm::identity_mat<T, n>() - K * H_k) * P;       //Updated covariance estimate
}
