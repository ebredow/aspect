/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/


#include <aspect/heating_model/shear_heating.h>


namespace aspect
{
  namespace HeatingModel
  {
    template <int dim>
    void
    ShearHeating<dim>::
    evaluate (const typename MaterialModel::Interface<dim>::MaterialModelInputs &material_model_inputs,
              const typename MaterialModel::Interface<dim>::MaterialModelOutputs &material_model_outputs,
              HeatingModel::HeatingModelOutputs &heating_model_outputs) const
    {
      Assert(heating_model_outputs.heating_source_terms.size() == material_model_inputs.position.size(),
             ExcMessage ("Heating outputs need to have the same number of entries as the material model inputs."));

      for (unsigned int q=0; q<heating_model_outputs.heating_source_terms.size(); ++q)
        {
          const Tensor<1,dim> gravity = this->get_gravity_model().gravity_vector(material_model_inputs.position[q]);

          heating_model_outputs.heating_source_terms[q] = 2.0 * material_model_outputs.viscosities[q] *
                                                          material_model_inputs.strain_rate[q] * material_model_inputs.strain_rate[q]
                                                          - (this->get_material_model().is_compressible()
                                                             ?
                                                             2./3. * material_model_outputs.viscosities[q]
                                                               * std::pow(material_model_outputs.compressibilities[q]
                                                                          * material_model_outputs.densities[q]
                                                                          * (material_model_inputs.velocity[q] * gravity),
                                                                          2)
                                                             :
                                                             0.0);
        }
    }

    template <int dim>
    void
    ShearHeating<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Shear heating");
        {

        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    ShearHeating<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Heating model");
      {
        prm.enter_subsection("Shear heating");
        {
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace HeatingModel
  {
    ASPECT_REGISTER_HEATING_MODEL(ShearHeating,
                                  "shear heating",
                                  "Implementation of a standard model for shear heating.")
  }
}
