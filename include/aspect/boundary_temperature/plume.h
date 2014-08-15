/*
  Copyright (C) 2014 by the authors of the ASPECT code.

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


#ifndef __aspect__boundary_temperature_plume_h
#define __aspect__boundary_temperature_plume_h

#include <aspect/boundary_temperature/interface.h>
#include <aspect/simulator_access.h>


namespace aspect
{
  namespace BoundaryTemperature
  {
    using namespace dealii;


    namespace internal
    {
      /**
       * GPlatesLookup handles all kinds of tasks around looking up a certain
       * velocity boundary condition from a gplates .gpml file. This class
       * keeps around the contents of two sets of files, corresponding to two
       * instances in time where GPlates provides us with data; the boundary
       * values at one particular time are interpolated between the two
       * currently loaded data sets.
       */
      template <int dim>
      class PlumeLookup
      {
        public:

          /**
           * Initialize all members and the two pointers referring to the
           * actual velocities. Also calculates any necessary rotation
           * parameters for a 2D model.
           */
          PlumeLookup(const std::string &filename,
                      const ConditionalOStream &pcout);

          /**
           * Check whether a file named @p filename exists.
           */
          bool fexists(const std::string &filename) const;

          /**
           * Returns the computed surface velocity in cartesian coordinates.
           * Takes as input the position and current time weight.
           *
           * @param position The current position to compute velocity
           * @param time_weight A weighting between the two current timesteps
           * n and n+1
           */
          Point<dim> plume_position(const double time) const;

        private:

          /**
           * Tables which contain the velocities
           */
          std::vector<Point<dim> > plume_positions;
          std::vector<double> times;

          unsigned int
          calculate_time_index(const double time) const;
      };
    }

    /**
     * A class that implements a temperature boundary condition for a box
     * geometry.
     *
     * @ingroup BoundaryTemperatures
     */
    template <int dim>
    class Plume : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Loads file.
         */
      virtual
        void
        initialize ();

        /**
         * A function that is called at the beginning of each time step. For
         * the current plugin, this function updates the current plume position.
         */
      virtual
        void
        update ();

        /**
         * Return the temperature that is to hold at a particular location on
         * the boundary of the domain. This function returns constant
         * temperatures at the left and right boundaries.
         *
         * @param geometry_model The geometry model that describes the domain.
         * This may be used to determine whether the boundary temperature
         * model is implemented for this geometry.
         * @param boundary_indicator The boundary indicator of the part of the
         * boundary of the domain on which the point is located at which we
         * are requesting the temperature.
         * @param location The location of the point at which we ask for the
         * temperature.
         */
        virtual
        double temperature (const GeometryModel::Interface<dim> &geometry_model,
                            const unsigned int                   boundary_indicator,
                            const Point<dim>                    &location) const;

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
         * Declare the parameters this class takes through input files. This
         * class declares the inner and outer boundary temperatures.
         */
        static
        void
        declare_parameters (ParameterHandler &prm);

        /**
         * Read the parameters this class declares from the parameter file.
         */
        virtual
        void
        parse_parameters (ParameterHandler &prm);

        /**
         * This is a workaround for the fact the the mesh refinement plugins
         * can not have an own plume_lookup object due to missing update() and
         * initialize() functions. It forwards the current plume position to
         * other plugins.
         */
        Point<dim>
        get_plume_position () const;


      private:
        double temperature_[2*dim];

        /**
         * Current plume position. Used to avoid looking up the plume position
         * for every quadrature point, since this is only time-dependent.
         */
        Point<dim> plume_position;

        /**
         * Pointer to an object that reads and processes data we get from
         * gplates files.
         */
        std_cxx1x::shared_ptr<internal::PlumeLookup<dim> > lookup;

        /**
         * Directory in which the plume file is present.
         */
        std::string data_directory;

        /**
         * Filename of plume file.
         */
        std::string plume_file_name;

        /**
         * Magnitude of the temperature anomaly
         */
        double T0;

        /**
         * Radius of the temperature anomaly
         */
        double R0;

    };
  }
}


#endif
