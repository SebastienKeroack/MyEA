namespace MyEA::Random
{
    template<>
    class Random_Real<adept::adouble> : public Random_Base
    {
        public:
            Random_Real(void) : Random_Base()
            {
                this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range));
            }

            Random_Real(unsigned int const seed_received,
                                        adept::adouble const minimum_range_received,
                                        adept::adouble const maximum_range_received) : Random_Base(seed_received)
            {
                double tmp_minimum_range(Cast_T(min(minimum_range_received, maximum_range_received))),
                       tmp_maximum_range(Cast_T(max(minimum_range_received, maximum_range_received)));

                this->_minimum_range = tmp_minimum_range;
                this->_maximum_range = tmp_maximum_range;

                this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range));
            }

            class Random_Real<adept::adouble>& operator=(class Random_Real<adept::adouble> const &ref_source_Class_Generator_Random_Real_received)
            {
                if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

                return(*this);
            }

            void Copy(class Random_Real<adept::adouble> const &ref_source_Class_Generator_Random_Real_received)
            {
                Random_Base::Copy(ref_source_Class_Generator_Random_Real_received);

                this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
                this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;

                this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
            }

            void Range(adept::adouble const minimum_range_received, adept::adouble const maximum_range_received)
            {
                double tmp_minimum_range(Cast_T(min(minimum_range_received, maximum_range_received))),
                       tmp_maximum_range(Cast_T(max(minimum_range_received, maximum_range_received)));

                if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

                this->_minimum_range = tmp_minimum_range;
                this->_maximum_range = tmp_maximum_range;

                this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range));
            }

            virtual void Clear(void)
            {
                this->Random_Base::Clear();

                this->Range(adept::adouble(0), adept::adouble(1));
            }

            virtual void Reset(void)
            {
                this->Random_Base::Reset();

                this->_uniform_real_distribution.reset();
            }

            adept::adouble Generate_Real(void) { return(this->_uniform_real_distribution(this->p_generator_mt19937)); }

        private:
            double _minimum_range = 0.0;
            double _maximum_range = 1.0;

            std::uniform_real_distribution<double> _uniform_real_distribution;
    };

    template<>
    class Random_Gaussian<adept::adouble> : public Random_Base
    {
        public:
            Random_Gaussian(void) : Random_Base()
            {
                this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std));
            }

            Random_Gaussian(unsigned int const seed_received,
                            adept::adouble const mean_received,
                            adept::adouble const std_received) : Random_Base(seed_received)
            {
                this->_mean = mean_received.value();
                this->_std = std_received.value();

                this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std));
            }

            class Random_Gaussian<adept::adouble>& operator=(class Random_Gaussian<adept::adouble> const &ref_source_Class_Generator_Random_Gaussian_received)
            {
                if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

                return(*this);
            }

            void Copy(class Random_Gaussian<adept::adouble> const &ref_source_Class_Generator_Random_Gaussian_received)
            {
                Random_Base::Copy(ref_source_Class_Generator_Random_Gaussian_received);

                this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
                this->_std = ref_source_Class_Generator_Random_Gaussian_received._std;

                this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
            }

            void Range(adept::adouble const mean_received, adept::adouble const std_received)
            {
                double const tmp_mean(mean_received.value()),
                             tmp_std(std_received.value());

                if(this->_mean == tmp_mean && this->_std == tmp_std) { return; }

                this->_mean = tmp_mean;
                this->_std = tmp_std;

                this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std));
            }

            virtual void Clear(void)
            {
                this->Random_Base::Clear();

                this->Range(0.0, 1.0);
            }

            virtual void Reset(void)
            {
                this->Random_Base::Reset();

                this->_normal_distribution.reset();
            }

            adept::adouble Generate_Gaussian(void) { return(this->_normal_distribution(this->p_generator_mt19937)); }

        private:
            double _mean = 0.0;
            double _std  = 1.0;

            std::normal_distribution<double> _normal_distribution;
    };

    template<>
    class Random_Bernoulli<adept::adouble> : public Random_Base
    {
        public:
            Random_Bernoulli<adept::adouble>::Random_Bernoulli(void) : Random_Base() { }

            Random_Bernoulli<adept::adouble>::Random_Bernoulli(unsigned int const seed_received) : Random_Base(seed_received) { }

            class Random_Bernoulli<adept::adouble>& Random_Bernoulli<adept::adouble>::operator=(class Random_Bernoulli<adept::adouble> const &ref_source_Class_Generator_Random_Bernoulli_received)
            {
                if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

                return(*this);
            }

            void Random_Bernoulli<adept::adouble>::Copy(class Random_Bernoulli<adept::adouble> const &ref_source_Class_Generator_Random_Bernoulli_received)
            {
                Random_Base::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

                this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;

                this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
            }

            void Random_Bernoulli<adept::adouble>::Probability(adept::adouble const probability_received)
            {
                double const tmp_probability(probability_received.value());

                if(tmp_probability < 0.0 || tmp_probability > 1.0) { return; }
                else if(this->_probability == tmp_probability) { return; }

                std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(tmp_probability);
                this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
            }

            void Random_Bernoulli<adept::adouble>::Clear(void)
            {
                this->Random_Base::Clear();

                this->_probability = 0.0;
            }

            void Random_Bernoulli<adept::adouble>::Reset(void)
            {
                this->Random_Base::Reset();

                this->_bernoulli_distribution.reset();
            }

            bool Random_Bernoulli<adept::adouble>::Generate_Bernoulli(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }

        private:
            double _probability = 0.0;

            std::bernoulli_distribution _bernoulli_distribution;
    };
}