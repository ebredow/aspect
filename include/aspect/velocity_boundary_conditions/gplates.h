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


#ifndef __aspect__velocity_boundary_conditions_gplates_h
#define __aspect__velocity_boundary_conditions_gplates_h

#include <aspect/velocity_boundary_conditions/interface.h>
#include <deal.II/base/std_cxx11/array.h>
#include <deal.II/base/point.h>
#include <deal.II/base/table.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>


namespace aspect
{
  namespace VelocityBoundaryConditions
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
      class GPlatesLookup
      {
        public:

          /**
           * Initialize all members and the two pointers referring to the
           * actual velocities. Also calculates any necessary rotation
           * parameters for a 2D model.
           */
          GPlatesLookup(const Tensor<1,2> &pointone,
                        const Tensor<1,2> &pointtwo);

          /**
           * Outputs the GPlates module information at model start. Need to be
           * separated from Constructor because at construction time the
           * SimulatorAccess is not initialized and only Rank 0 should give
           * the screen output.
           */
          void screen_output(const Tensor<1,2> &surface_point_one,
                             const Tensor<1,2> &surface_point_two,
                             const ConditionalOStream &pcout) const;

          /**
           * Loads a gplates .gpml velocity file. Throws an exception if the
           * file does not exist.
           */
          void load_file(const std::string &filename);

          /**
           * Returns the computed surface velocity in cartesian coordinates.
           * Takes as input the position.
           *
           * @param position The current position to compute velocity
           */
          Tensor<1,dim> surface_velocity(const Point<dim> &position) const;

        private:
          /**
           * Interpolation functions to access the velocities.
           */
          std::vector<Functions::InterpolatedUniformGridData<2> *> velocities;

          /**
           * Min and Max coordinates in data file
           */
          std_cxx11::array<std::pair<double,double>,2> grid_extent;

          /**
           * Distances between adjacent point in the Lat/Long grid
           */
          double delta_phi, delta_theta;

          /**
           * The matrix, which describes the rotation by which a 2D model
           * needs to be transformed to a plane that contains the origin and
           * the two user prescribed points. Is not used for 3D.
           */
          Tensor<2,3> rotation_matrix;

          /**
           * A function that returns the rotated vector r' that results out of
           * a rotation from vector r around a specified rotation_axis by an
           * defined angle
           */
          Tensor<1,3>
          rotate_around_axis (const Tensor<1,3> &position,
                              const Tensor<1,3> &rotation_axis,
                              const double angle) const;

          /**
           * A function that returns the corresponding paraview angles for a
           * rotation described by a rotation matrix. These differ from the
           * usually used euler angles by assuming a rotation around the
           * coordinate axes in the order y-x-z (instead of the often used
           * z-x-z)
           */
          std_cxx11::array<double,3>
          angles_from_matrix (const Tensor<2,3> &rotation_matrix) const;

          /**
           * A function that returns the corresponding rotation axis/angle for
           * a rotation described by a rotation matrix.
           */
          double
          rotation_axis_from_matrix (Tensor<1,3> &rotation_axis,
                                     const Tensor<2,3> &rotation_matrix) const;

          /**
           * A function that returns the corresponding euler angles for a
           * rotation described by rotation axis and angle.
           */
          Tensor<2,3>
          rotation_matrix_from_axis (const Tensor<1,3> &rotation_axis,
                                     const double rotation_angle) const;

          /**
           * Convert a tensor of rank 1 and dimension in to rank 1 and
           * dimension out. If $out < in$ the last elements will be discarded,
           * if $out > in$ zeroes will be appended to fill the tensor.
           */
          template <int in, int out>
          Tensor<1,out> convert_tensor (const Tensor<1,in> &old_tensor) const;

          /**
           * Return the cartesian coordinates of a spherical surface position
           * defined by theta (polar angle, not geographical latitude) and
           * phi.
           */
          Tensor<1,3>
          cartesian_surface_coordinates(const Tensor<1,3> &sposition) const;

          /**
           * Returns cartesian velocities calculated from surface velocities
           * and position in spherical coordinates
           *
           * @param s_velocities Surface velocities in spherical coordinates
           * (theta, phi)
           * @param s_position Position in spherical coordinates
           * // alt:(theta,phi,radius)
           * (radius,phi,theta)
           */
          Tensor<1,3> sphere_to_cart_velocity(const Tensor<1,2> &s_velocities,
                                              const std_cxx11::array<double,3> &s_position) const;


          /**
           * Returns a velocity vector that is rotated to be tangential to the
           * sphere surface at point position
           *
           * @param data_position Position of the velocity data point
           * @param point_position Position of the current evaluation point to
           * which the velocity will be rotated
           * @param data_velocity Unrotated velocity vector
           */
          Tensor<1,3>
          rotate_grid_velocity(const Tensor<1,3> &data_position,
                               const Tensor<1,3> &point_position,
                               const Tensor<1,3> &data_velocity) const;

          /**
           * Returns the position (cartesian or spherical depending on last
           * argument) of a data point with a given theta,phi index.
           */
          Tensor<1,3>
          get_grid_point_position(const unsigned int theta_index,
                                  const unsigned int phi_index,
                                  const bool cartesian) const;

          /**
           * Returns the arc distance of two points on a sphere surface.
           */
          double
          arc_distance(const Tensor<1,3> position_1, const Tensor<1,3> position_2) const;

          /**
           * Check whether the gpml file was created by GPlates1.4 or later.
           * We need to know this, because the mesh has changed its longitude
           * origin from 0 to -180 degrees and we need to correct for this.
           */
          bool
          gplates_1_4_or_higher(const boost::property_tree::ptree &pt) const;
      };
    }

    /**
     * A class that implements prescribed velocity boundary conditions
     * determined from a GPlates input files.
     *
     * @ingroup VelocityBoundaryConditionsModels
     */
    template <int dim>
    class GPlates : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Empty Constructor.
         */
        GPlates ();

        /**
         * Return the boundary velocity as a function of position. For the
         * current class, this function returns value from gplates.
         */
        Tensor<1,dim>
        boundary_velocity (const Point<dim> &position) const;

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        virtual
        void
        initialize ();

        /**
         * A function that is called at the beginning of each time step. For
         * the current plugin, this function loads the next velocity files if
         * necessary and outputs a warning if the end of the set of velocity
         * files is reached.
         */
        virtual
        void
        update ();

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
         * A variable that stores the currently used data file of a series. It
         * gets updated if necessary by update().
         */
        int  current_file_number;

        /**
         * Time from which on the data file with number 'First data file
         * number' is used as boundary condition. Previous to this time, 0 is
         * returned for every field. Depending on the setting of the global
         * 'Use years in output instead of seconds' flag in the input file,
         * this number is either interpreted as seconds or as years."
         */
        double first_data_file_model_time;

        /**
         * Number of the first data file to be loaded when the model time is
         * larger than 'First data file model time'.
         */
        int first_data_file_number;

        /**
         * In some cases the boundary files are not numbered in increasing but
         * in decreasing order (e.g. 'Ma BP'). If this flag is set to 'True'
         * the plugin will first load the file with the number 'First data
         * file number' and decrease the file number during the model run.
         */
        bool decreasing_file_order;

        /**
         * Time in model units (depends on other model inputs) between two
         * data files.
         */
        double data_file_time_step;

        /**
         * Weight between data file n and n+1 while the current time is
         * between the two values t(n) and t(n+1).
         */
        double time_weight;

        /**
         * State whether we have time_dependent boundary conditions. Switched
         * off after finding no more data files to suppress attempts to read
         * in new files.
         */
        bool time_dependent;

        /**
         * Directory in which the gplates velocity are present.
         */
        std::string data_directory;

        /**
         * First part of filename of velocity files. The files have to have
         * the pattern velocity_file_name.n.gpml where n is the number of the
         * current timestep (starts from 0).
         */
        std::string velocity_file_name;

        /**
         * Scale the velocity boundary condition by a scalar factor.
         */
        double scale_factor;

        /**
         * Two user defined points that prescribe the plane from which the 2D
         * model takes the velocity boundary condition. One can think of this,
         * as if the model is lying in this plane although no actual model
         * coordinate is changed. The strings need to have the format "a,b"
         * where a and b are doubles and define theta and phi on a sphere.
         */
        std::string point1;
        std::string point2;

        /**
         * Parsed user input of point1 and point2
         */
        Tensor<1,2> pointone;
        Tensor<1,2> pointtwo;

        /**
         * Determines the depth of the lithosphere. The user might want to apply
         * the GPlates velocities not only at the surface of the model, but also
         * in the whole lithosphere.
         */
        double lithosphere_thickness;

        /**
         * Pointer to an object that reads and processes data we get from
         * gplates files.
         */
        std_cxx11::shared_ptr<internal::GPlatesLookup<dim> > lookup;

        /**
         * Pointer to an object that reads and processes data we get from
         * gplates files. This saves the previous data time step.
         */
        std_cxx11::shared_ptr<internal::GPlatesLookup<dim> > old_lookup;

        /**
         * Handles the update of the velocity data in lookup.
         */
        void
        update_data (const bool reload_both_files);

        /**
         * Handles settings and user notification in case the time-dependent
         * part of the boundary condition is over.
         */
        void
        end_time_dependence ();

        /**
         * Create a filename out of the name template.
         */
        std::string
        create_filename (const int timestep) const;
    };
  }
}


#endif
