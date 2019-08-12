#include "stdafx.hpp"

// This.
#include <Random/Base.hpp>

namespace MyEA::Random
{
    Base::Base(unsigned int const seed_received)
    {
        this->Seed(seed_received);
    }

    class Base& Base::operator=(class Base const &ref_source_Class_Generator_Random_received)
    {
        if(&ref_source_Class_Generator_Random_received != this) { this->Copy(ref_source_Class_Generator_Random_received); }

        return(*this);
    }

    Base::~Base(void) { }

    void Base::Copy(class Base const &ref_source_Class_Generator_Random_received)
    {
        this->p_generator_mt19937 = ref_source_Class_Generator_Random_received.p_generator_mt19937;

        this->p_seed = ref_source_Class_Generator_Random_received.p_seed;
    }

    void Base::Seed(unsigned int const seed_received)
    {
        this->p_generator_mt19937.seed(seed_received);

        this->p_seed = seed_received;
    }

    void Base::Clear(void) { this->p_seed = 5413u; }

    void Base::Reset(void) { this->Seed(this->p_seed); }
}
