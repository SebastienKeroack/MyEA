namespace MyEA
{
    namespace Common
    {
        template<>
        class Class_Generator_Random_Real<adept::adouble> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Real(void) : Class_Generator_Random() { this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range)); }
                Class_Generator_Random_Real(unsigned int const seed_received,
                                                                adept::adouble const minimum_range_received,
                                                                adept::adouble const maximum_range_received) : Class_Generator_Random(seed_received)
                {
                    adept::adouble tmp_minimum_range(min(minimum_range_received, maximum_range_received)),
                                          tmp_maximum_range(max(minimum_range_received, maximum_range_received));

                    this->_minimum_range = tmp_minimum_range;
                    this->_maximum_range = tmp_maximum_range;
            
                    this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range));
                }
                
                class Class_Generator_Random_Real<adept::adouble>& operator=(class Class_Generator_Random_Real<adept::adouble> const &ref_source_Class_Generator_Random_Real_received)
                {
                    if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

                    return(*this);
                }
                
                void Copy(class Class_Generator_Random_Real<adept::adouble> const &ref_source_Class_Generator_Random_Real_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Real_received);

                    this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
                    this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;
            
                    this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
                }
                void Range(adept::adouble const minimum_range_received, adept::adouble const maximum_range_received)
                {
                    adept::adouble tmp_minimum_range(min(minimum_range_received, maximum_range_received)),
                                          tmp_maximum_range(max(minimum_range_received, maximum_range_received));
            
                    if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

                    this->_minimum_range = tmp_minimum_range;
                    this->_maximum_range = tmp_maximum_range;
            
                    this->_uniform_real_distribution.param(typename std::uniform_real_distribution<double>::param_type(this->_minimum_range, this->_maximum_range));
                }
                virtual void Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->Range(adept::adouble(0), adept::adouble(1));
                }
                virtual void Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_uniform_real_distribution.reset();
                }

                adept::adouble Generate_Real(void) { return(this->_uniform_real_distribution(this->p_generator_mt19937)); }

            private:
                double _minimum_range = 0;
                double _maximum_range = 1;

                std::uniform_real_distribution<double> _uniform_real_distribution;
        };
        
        template<>
        class Class_Generator_Random_Gaussian<adept::adouble> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Gaussian(void) : Class_Generator_Random() { this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std)); }
                Class_Generator_Random_Gaussian(unsigned int const seed_received,
                                                                       adept::adouble const mean_received,
                                                                       adept::adouble const std_received) : Class_Generator_Random(seed_received)
                {
                    this->_mean = mean_received;
                    this->_std = std_received;
                    
                    this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std));
                }
                
                class Class_Generator_Random_Gaussian<adept::adouble>& operator=(class Class_Generator_Random_Gaussian<adept::adouble> const &ref_source_Class_Generator_Random_Gaussian_received)
                {
                    if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

                    return(*this);
                }
                
                void Copy(class Class_Generator_Random_Gaussian<adept::adouble> const &ref_source_Class_Generator_Random_Gaussian_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Gaussian_received);

                    this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
                    this->_std = ref_source_Class_Generator_Random_Gaussian_received._std;
                    
                    this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
                }
                void Range(adept::adouble const mean_received, adept::adouble const std_received)
                {
                    if(this->_mean == mean_received && this->_std == std_received) { return; }

                    this->_mean = mean_received;
                    this->_std = std_received;
                    
                    this->_normal_distribution.param(typename std::normal_distribution<double>::param_type(this->_mean, this->_std));
                }
                virtual void Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->Range(0.0, 1.0);
                }
                virtual void Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_normal_distribution.reset();
                }

                adept::adouble Generate_Gaussian(void) { return(this->_normal_distribution(this->p_generator_mt19937)); }

            private:
                adept::adouble _mean = 0;
                adept::adouble _std = 1;

                std::normal_distribution<double> _normal_distribution;
        };
        
        template<>
        class Class_Generator_Random_Bernoulli<adept::adouble> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Bernoulli<adept::adouble>::Class_Generator_Random_Bernoulli(void) : Class_Generator_Random() { }
                
                Class_Generator_Random_Bernoulli<adept::adouble>::Class_Generator_Random_Bernoulli(unsigned int const seed_received) : Class_Generator_Random(seed_received) { }
                
                class Class_Generator_Random_Bernoulli<adept::adouble>& Class_Generator_Random_Bernoulli<adept::adouble>::operator=(class Class_Generator_Random_Bernoulli<adept::adouble> const &ref_source_Class_Generator_Random_Bernoulli_received)
                {
                    if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

                    return(*this);
                }

                void Class_Generator_Random_Bernoulli<adept::adouble>::Copy(class Class_Generator_Random_Bernoulli<adept::adouble> const &ref_source_Class_Generator_Random_Bernoulli_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

                    this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;
                    
                    this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
                }
                
                void Class_Generator_Random_Bernoulli<adept::adouble>::Probability(adept::adouble const probability_received)
                {
                    if(probability_received < 0.0 || probability_received > 1.0) { return; }
                    else if(this->_probability == probability_received) { return; }
                    
                    std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(probability_received);
                    this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
                }
                
                void Class_Generator_Random_Bernoulli<adept::adouble>::Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->_probability = 0.0;
                }
                
                void Class_Generator_Random_Bernoulli<adept::adouble>::Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_bernoulli_distribution.reset();
                }
                
                bool Class_Generator_Random_Bernoulli<adept::adouble>::Generate_Bernoulli(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }
                
            private:
                double _probability = 0.0;

                std::bernoulli_distribution _bernoulli_distribution;
        };
    }
}