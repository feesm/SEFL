/*! @file ekf.hpp
 *
 *  @author Moritz Fees
 *  @date 2024-09-03
 *
 */

#include <boost/qvm/lite.hpp>


namespace sefl
{

    /*! @brief Extended Kalman Filter
     *  
     *  @tparam T Type used for variables
     *  @tparam n Size of state estimate
     *
     */
    template<class T, int n>
    class ekf
    {
        public:

            /*! @brief Initializes the Extended Kalman Filter
             *
             *  @param P_0 Initial Covariance estimate
             *  @param x_0 Initial state estimate
             *
             */
            ekf(boost::qvm::mat<T, n, n> P_0 = boost::qvm::identity_mat<T, n>(),
                    boost::qvm::vec<T, n> x_0 = boost::qvm::zero_vec<T, n>()):
                x(x_0), P(P_0) {}

            /*! @brief Predicts the new state with the control vector u
             *
             *  @param u Control vector
             *  @param var_u Variance of the control vector
             *  @param dt Time since last prediction
             *  @param f State transition function
             *  @param F Jacobian of state transition function
             *  @tparam m Size of control vector
             *
             */
            template<int m> 
            void predict(boost::qvm::vec<T, m> u, boost::qvm::vec<T, m> var_u, T dt,
                    boost::qvm::vec<T, n> (*f)(boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u),
                    boost::qvm::mat<T, n, n> (*F) (boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u));
            
            /*! @brief Updates the state estimate with new observation
             *  
             *  @param z Observation vector
             *  @param R Covariance of observation
             *  @param h State observation function
             *  @param H Jacobian of state observation function
             *  @tparam m size of observation vector
             *
             */
            template<int m>
            void update(boost::qvm::vec<T, m> z, boost::qvm::mat<T, n, n> R,
                    boost::qvm::vec<T, m> (*h) (boost::qvm::vec<T, n> x),
                    boost::qvm::mat<T, m, m> (*H) (boost::qvm::vec<T, n> x));

        private:

            /*! @brief State estimate
             */
            boost::qvm::vec<T, n> x;

            /*! @brief Covariance estimate
             */
            boost::qvm::mat<T, n, n> P;            
    };
}

template<class T, int n>
template<int m>
void sefl::ekf<T, n>::predict(boost::qvm::vec<T, m> u, boost::qvm::vec<T, m> var_u, T dt,
                    boost::qvm::vec<T, n> (*f)(boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u),
                    boost::qvm::mat<T, n, n> (*F) (boost::qvm::vec<T, n> x, boost::qvm::vec<T, m> u))
{
    boost::qvm::mat<T, n, n> Q = f(x, var_u);
    boost::qvm::mat<T, n, n> F_k= F(x,u);
    P += (F_k * P + P * transposed(F_k) + Q) * dt;              //Predicted covariance estimate
    x += f(x,u) * dt;                                           //Predicted state estimate
}

template<class T, int n>
template<int m>
void sefl::ekf<T, n>::update(boost::qvm::vec<T, m> z, boost::qvm::mat<T, n, n> R,
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
