/*
  Copyright (C) 2011 - 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__boundary_temperature_ascii_data_h
#define __aspect__boundary_temperature_ascii_data_h

#include <aspect/boundary_temperature/interface.h>
#include <aspect/velocity_boundary_conditions/ascii_data.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace BoundaryTemperature
  {
    using namespace dealii;

    /**
     * A class that implements prescribed data boundary conditions
     * determined from a AsciiData input file.
     *
     * @ingroup BoundaryTemperatures
     */
    template <int dim>
    class AsciiData : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        AsciiData ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        virtual
        void
        initialize ();

        /**
         * A function that is called at the beginning of each time step. For
         * the current plugin, this function loads the next data files if
         * necessary and outputs a warning if the end of the set of data
         * files is reached.
         */
        virtual
        void
        update ();

        /**
         * Return the boundary temperature as a function of position. For the
         * current class, this function returns value from the text files.
         */
        double
        temperature (const GeometryModel::Interface<dim> &geometry_model,
                     const unsigned int                   boundary_indicator,
                     const Point<dim> &position) const;

        /**
         * Return the minimal the temperature on that part of the boundary on
         * which Dirichlet conditions are posed.
         *
         * This value is used in computing dimensionless numbers such as the
         * Nusselt number indicating heat flux.
         */
        virtual
        double minimal_temperature (const std::set<types::boundary_id> &fixed_boundary_ids) const;

        /**
         * Return the maximal the temperature on that part of the boundary on
         * which Dirichlet conditions are posed.
         *
         * This value is used in computing dimensionless numbers such as the
         * Nusselt number indicating heat flux.
         */
        virtual
        double maximal_temperature (const std::set<types::boundary_id> &fixed_boundary_ids) const;


        /**
         * Declare the parameters this class takes through input files.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        void
        parse_parameters (ParameterHandler &prm);

      private:
        /**
         * A variable that stores the currently used data file of a
         * series. It gets updated if necessary by update_data().
         */
        int  current_file_number;

        /**
         * Time from which on the data file with number 'First data
         * file number' is used as boundary condition. Previous to this
         * time, a temperature of 0 is returned. Depending on the setting
         * of the global 'Use years in output instead of seconds' flag
         * in the input file, this number is either interpreted as seconds or as years."
         */
        double first_data_file_model_time;

        /**
         * Number of the first data file to be loaded when the model time
         * is larger than 'First data file model time'.
         */
        int first_data_file_number;

        /**
         * In some cases the boundary files are not numbered in increasing
         * but in decreasing order (e.g. 'Ma BP'). If this flag is set to
         * 'True' the plugin will first load the file with the number
         * 'First data file number' and decrease the file number during
         * the model run.
         */
        bool decreasing_file_order;

        /**
         * Directory in which the data files are present.
         */
        std::string data_directory;

        /**
         * Filename of data file. The file names can contain
         * the specifiers %s and/or %c (in this order), meaning the name of the
         * boundary as provided by the geometry model and the number of the
         * data file time step.
         */
        std::string data_file_name;

        /**
         * Time in model units (depends on other model inputs) between two
         * data files.
         */
        double data_file_time_step;

        /**
         * Number of grid points in data file. To set this parameter in the
         * parameter file is optional, if a header is provided in the first
         * file.
         */
        std_cxx11::array<unsigned int,3> data_points;

        /**
         * Weight between data file n and n+1 while the current time is
         * between the two values t(n) and t(n+1).
         */
        double time_weight;

        /**
         * State whether we have time_dependent boundary conditions. Switched
         * off after finding no more data files to suppress attempts to
         * read in new files.
         */
        bool time_dependent;

        /**
         * Scale the data boundary condition by a scalar factor.
         */
        double scale_factor;

        /**
         * Map between the boundary id and an object that reads and processes
         * data we get from text files.
         */
        std::map<types::boundary_id,
                  std_cxx11::shared_ptr<VelocityBoundaryConditions::internal::AsciiDataLookup<dim,dim-1> > > lookups;

        /**
         * Handles the update of the data in lookup.
         */
        void
        update_data (const types::boundary_id boundary_id);

        /**
         * Handles settings and user notification in case the time-dependent
         * part of the boundary condition is over.
         */
        void
        end_time_dependence (const int timestep,
                             const types::boundary_id boundary_id);

        /**
         * Create a filename out of the name template.
         */
        std::string
        create_filename (const int timestep,
                         const types::boundary_id boundary_id) const;
    };
  }
}


#endif
