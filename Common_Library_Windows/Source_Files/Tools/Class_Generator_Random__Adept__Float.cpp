namespace MyEA
{
    namespace Common
    {
        template<>
        class Class_Generator_Random_Real<adept::afloat> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Real(void) : Class_Generator_Random() { this->_uniform_real_distribution.param(typename std::uniform_real_distribution<float>::param_type(this->_minimum_range, this->_maximum_range)); }
                Class_Generator_Random_Real(unsigned int const seed_received,
                                            adept::afloat const minimum_range_received,
                                            adept::afloat const maximum_range_received) : Class_Generator_Random(seed_received)
                {
                    float const tmp_minimum_range(Cast_T(min(minimum_range_received, maximum_range_received))),
                                tmp_maximum_range(Cast_T(max(minimum_range_received, maximum_range_received)));

                    this->_minimum_range = tmp_minimum_range;
                    this->_maximum_range = tmp_maximum_range;
            
                    this->_uniform_real_distribution.param(typename std::uniform_real_distribution<float>::param_type(this->_minimum_range, this->_maximum_range));
                }
                
                class Class_Generator_Random_Real<adept::afloat>& operator=(class Class_Generator_Random_Real<adept::afloat> const &ref_source_Class_Generator_Random_Real_received)
                {
                    if(&ref_source_Class_Generator_Random_Real_received != this) { this->Copy(ref_source_Class_Generator_Random_Real_received); }

                    return(*this);
                }
                
                void Copy(class Class_Generator_Random_Real<adept::afloat> const &ref_source_Class_Generator_Random_Real_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Real_received);

                    this->_minimum_range = ref_source_Class_Generator_Random_Real_received._minimum_range;
                    this->_maximum_range = ref_source_Class_Generator_Random_Real_received._maximum_range;
            
                    this->_uniform_real_distribution = ref_source_Class_Generator_Random_Real_received._uniform_real_distribution;
                }
                void Range(adept::afloat const minimum_range_received, adept::afloat const maximum_range_received)
                {
                    float const tmp_minimum_range(Cast_T(min(minimum_range_received, maximum_range_received))),
                                tmp_maximum_range(Cast_T(max(minimum_range_received, maximum_range_received)));
            
                    if(this->_minimum_range == tmp_minimum_range && this->_maximum_range == tmp_maximum_range) { return; }

                    this->_minimum_range = tmp_minimum_range;
                    this->_maximum_range = tmp_maximum_range;
            
                    this->_uniform_real_distribution.param(typename std::uniform_real_distribution<float>::param_type(this->_minimum_range, this->_maximum_range));
                }
                virtual void Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->Range(adept::afloat(0), adept::afloat(1));
                }
                virtual void Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_uniform_real_distribution.reset();
                }

                adept::afloat Generate_Real(void) { return(this->_uniform_real_distribution(this->p_generator_mt19937)); }

            private:
                float _minimum_range  = 0.0f;
                float _maximum_range = 1.0f;

                std::uniform_real_distribution<float> _uniform_real_distribution;
        };
        
        template<>
        class Class_Generator_Random_Gaussian<adept::afloat> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Gaussian(void) : Class_Generator_Random() { this->_normal_distribution.param(typename std::normal_distribution<float>::param_type(this->_mean, this->_std)); }
                Class_Generator_Random_Gaussian(unsigned int const seed_received,
                                                adept::afloat const mean_received,
                                                adept::afloat const std_received) : Class_Generator_Random(seed_received)
                {
                    this->_mean = mean_received.value();
                    this->_std = std_received.value();
                    
                    this->_normal_distribution.param(typename std::normal_distribution<float>::param_type(this->_mean, this->_std));
                }
                
                class Class_Generator_Random_Gaussian<adept::afloat>& operator=(class Class_Generator_Random_Gaussian<adept::afloat> const &ref_source_Class_Generator_Random_Gaussian_received)
                {
                    if(&ref_source_Class_Generator_Random_Gaussian_received != this) { this->Copy(ref_source_Class_Generator_Random_Gaussian_received); }

                    return(*this);
                }
                
                void Copy(class Class_Generator_Random_Gaussian<adept::afloat> const &ref_source_Class_Generator_Random_Gaussian_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Gaussian_received);

                    this->_mean = ref_source_Class_Generator_Random_Gaussian_received._mean;
                    this->_std = ref_source_Class_Generator_Random_Gaussian_received._std;
                    
                    this->_normal_distribution = ref_source_Class_Generator_Random_Gaussian_received._normal_distribution;
                }
                void Range(adept::afloat const mean_received, adept::afloat const std_received)
                {
                    float const tmp_mean(mean_received.value()),
                                tmp_std(std_received.value());
                    
                    if(this->_mean == tmp_mean && this->_std == tmp_std) { return; }

                    this->_mean = tmp_mean;
                    this->_std = tmp_std;
                    
                    this->_normal_distribution.param(typename std::normal_distribution<float>::param_type(this->_mean, this->_std));
                }
                virtual void Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->Range(0.0f, 1.0f);
                }
                virtual void Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_normal_distribution.reset();
                }

                adept::afloat Generate_Gaussian(void) { return(this->_normal_distribution(this->p_generator_mt19937)); }

            private:
                float _mean = 0.0f;
                float _std = 1.0f;

                std::normal_distribution<float> _normal_distribution;
        };
        
        template<>
        class Class_Generator_Random_Bernoulli<adept::afloat> : public Class_Generator_Random
        {
            public:
                Class_Generator_Random_Bernoulli<adept::afloat>::Class_Generator_Random_Bernoulli(void) : Class_Generator_Random() { }
                
                Class_Generator_Random_Bernoulli<adept::afloat>::Class_Generator_Random_Bernoulli(unsigned int const seed_received) : Class_Generator_Random(seed_received) { }
                
                class Class_Generator_Random_Bernoulli<adept::afloat>& Class_Generator_Random_Bernoulli<adept::afloat>::operator=(class Class_Generator_Random_Bernoulli<adept::afloat> const &ref_source_Class_Generator_Random_Bernoulli_received)
                {
                    if(&ref_source_Class_Generator_Random_Bernoulli_received != this) { this->Copy(ref_source_Class_Generator_Random_Bernoulli_received); }

                    return(*this);
                }

                void Class_Generator_Random_Bernoulli<adept::afloat>::Copy(class Class_Generator_Random_Bernoulli<adept::afloat> const &ref_source_Class_Generator_Random_Bernoulli_received)
                {
                    Class_Generator_Random::Copy(ref_source_Class_Generator_Random_Bernoulli_received);

                    this->_probability = ref_source_Class_Generator_Random_Bernoulli_received._probability;
                    
                    this->_bernoulli_distribution = ref_source_Class_Generator_Random_Bernoulli_received._bernoulli_distribution;
                }
                
                void Class_Generator_Random_Bernoulli<adept::afloat>::Probability(adept::afloat const probability_received)
                {
                    float const tmp_probability(probability_received.value());

                    if(tmp_probability < 0.0f || tmp_probability > 1.0f) { return; }
                    else if(this->_probability == tmp_probability) { return; }
                    
                    std::bernoulli_distribution::param_type tmp_bernoulli_distribution_param_type(tmp_probability);
                    this->_bernoulli_distribution.param(tmp_bernoulli_distribution_param_type);
                }
                
                void Class_Generator_Random_Bernoulli<adept::afloat>::Clear(void)
                {
                    this->Class_Generator_Random::Clear();

                    this->_probability = 0.0f;
                }
                
                void Class_Generator_Random_Bernoulli<adept::afloat>::Reset(void)
                {
                    this->Class_Generator_Random::Reset();

                    this->_bernoulli_distribution.reset();
                }
                
                bool Class_Generator_Random_Bernoulli<adept::afloat>::Generate_Bernoulli(void) { return(this->_bernoulli_distribution(this->p_generator_mt19937)); }
                
            private:
                double _probability = 0.0f;

                std::bernoulli_distribution _bernoulli_distribution;
        };
    }
}