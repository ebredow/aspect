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


#include <aspect/global.h>
#include <aspect/velocity_boundary_conditions/plume_box.h>
#include <aspect/geometry_model/box.h>

#include <aspect/utilities.h>

#include <deal.II/base/parameter_handler.h>


namespace aspect
{
  namespace VelocityBoundaryConditions
  {
    template <int dim>
    PlumeBox<dim>::PlumeBox ()
      :
      current_file_number(0),
      first_data_file_model_time(0.0),
      first_data_file_number(0),
      decreasing_file_order(false),
      boundary_id(numbers::invalid_boundary_id),
      data_file_time_step(0.0),
      time_weight(0.0),
      time_dependent(true),
      scale_factor(1.0),
      lookup()
    {}


    template <int dim>
    void
    PlumeBox<dim>::initialize ()
    {
      // verify that the geometry is in fact a box since only
      // for this geometry do we know for sure what boundary indicators it
      // uses and what they mean
      Assert (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model())
              != 0,
              ExcMessage ("This boundary model is only implemented if the geometry is "
                          "in fact a box."));

      // First find the boundary_id this plugin is responsible for
      const std::map<types::boundary_id,std_cxx1x::shared_ptr<VelocityBoundaryConditions::Interface<dim> > >
        bvs = this->get_prescribed_velocity_boundary_conditions();
      for (typename std::map<types::boundary_id,std_cxx1x::shared_ptr<VelocityBoundaryConditions::Interface<dim> > >::const_iterator
           p = bvs.begin();
           p != bvs.end(); ++p)
        {
          if (p->second.get() == this)
            boundary_id = p->first;
        }
      AssertThrow(boundary_id != numbers::invalid_boundary_id,
                  ExcMessage("Did not find the boundary indicator for the prescribed data plugin."));


      // Initialize the bottom plume influx
      plume_lookup.reset(new BoundaryTemperature::internal::PlumeLookup<dim>(plume_data_directory+plume_file_name,
                                                                             this->get_pcout()));

      // Initialize the surface plate velocities
      const GeometryModel::Box<dim> *geometry_model = dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model());
      const Point<dim> grid_extent = geometry_model->get_extents();

      // Set the first file number and load the first files
      current_file_number = first_data_file_number;

      const int next_file_number =
          (decreasing_file_order) ?
              current_file_number - 1
              :
              current_file_number + 1;

      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
        {
          surface_lookup.reset(new internal::BoxPlatesLookup<dim>(surface_data_directory+surface_velocity_file_name,
                                                                  grid_extent,
                                                                  x_step,
                                                                  y_step,
                                                                  interpolation_width,
                                                                  this->get_pcout(),
                                                                  surface_time_scale_factor,
                                                                  surface_scale_factor));

          surface_lookup->load_file(create_surface_filename (current_file_number),
                                    this->get_pcout());
        }
      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
        {
          // Load side and bottom boundaries, but only if this is not the top boundary
          lookup.reset(new internal::AsciiDataLookup<dim,dim-1>(data_points,
                                                                this->get_geometry_model(),
                                                                dim,
                                                                scale_factor,
                                                                boundary_id));

          lookup->screen_output(this->get_pcout());

          this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
              << create_filename (current_file_number) << "." << std::endl << std::endl;
          lookup->load_file(create_filename (current_file_number));
        }


      // If the boundary condition is constant, switch
      // off time_dependence immediately. This also sets time_weight to 1.0.
      // If not, also load the second file for interpolation.
      if (create_filename (current_file_number) == create_filename (current_file_number+1))
        end_time_dependence (current_file_number);
      else
        {
          try
          {
              if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
                surface_lookup->load_file(create_surface_filename (current_file_number),
                                          this->get_pcout());

              if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
                {
                  this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                      << create_filename (next_file_number) << "." << std::endl << std::endl;
                  lookup->load_file(create_filename (next_file_number));
                }
          }
          catch (...)
          {
              end_time_dependence (current_file_number);
          }
        }
    }


    template <int dim>
    std::string
    PlumeBox<dim>::create_filename (const int filenumber) const
    {
      std::string templ = data_directory+data_file_name;
      const int size = templ.length();
      char *filename = (char *) (malloc ((size + 10) * sizeof(char)));
      const std::string boundary_id_name = this->get_geometry_model().translate_id_to_symbol_name(boundary_id);
      snprintf (filename, size + 10, templ.c_str (), boundary_id_name.c_str(),filenumber);
      std::string str_filename (filename);
      free (filename);
      return str_filename;
    }

    template <int dim>
    std::string
    PlumeBox<dim>::create_surface_filename (const int filenumber) const
    {
      std::string templ = surface_data_directory+surface_id_file_names;
      const int size = templ.length();
      char *filename = (char *) (malloc ((size + 10) * sizeof(char)));
      snprintf (filename, size + 10, templ.c_str (),filenumber);
      std::string str_filename (filename);
      free (filename);
      return str_filename;
    }


    template <int dim>
    void
    PlumeBox<dim>::update ()
    {
      Interface<dim>::update ();

      // First update the plume position, the coordinate system of the plume data files centers around 0
      // but that is the origin of our box, so we add half of the extent to align the two systems
      Point<dim> half_extents = (dynamic_cast<const GeometryModel::Box<dim> *>(&this->get_geometry_model()))->get_extents();
      half_extents(dim-1) = 0;
      half_extents /= 2.0;
      plume_position = plume_lookup->plume_position(this->get_time()) + half_extents;

      if (time_dependent && (this->get_time() - first_data_file_model_time >= 0.0))
        {
          // whether we need to update our data files. This looks so complicated
          // because we need to catch increasing and decreasing file orders and all
          // possible first_data_file_model_times and first_data_file_numbers.
          const bool need_update =
              static_cast<int> ((this->get_time() - first_data_file_model_time) / data_file_time_step)
              > std::abs(current_file_number - first_data_file_number);

          if (need_update)
            update_data();

          if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
            surface_lookup->update(this->get_time() - first_data_file_model_time);

          if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
            time_weight = (this->get_time() - first_data_file_model_time) / data_file_time_step
                          - std::abs(current_file_number - first_data_file_number);

          Assert ((0 <= time_weight) && (time_weight <= 1),
                  ExcMessage (
                    "Error in set_current_time. Time_weight has to be in [0,1]"));
        }
    }


    template <int dim>
    void
    PlumeBox<dim>::update_data ()
    {
      // The last file, which was tried to be loaded was
      // number current_file_number +/- 1, because current_file_number
      // is the file older than the current model time
      const int old_file_number =
          (decreasing_file_order) ?
              current_file_number - 1
              :
              current_file_number + 1;

      //Calculate new file_number
      current_file_number =
          (decreasing_file_order) ?
              first_data_file_number
                - static_cast<unsigned int> ((this->get_time() - first_data_file_model_time) / data_file_time_step)
              :
              first_data_file_number
                + static_cast<unsigned int> ((this->get_time() - first_data_file_model_time) / data_file_time_step);

      const int next_file_number =
          (decreasing_file_order) ?
              current_file_number - 1
              :
              current_file_number + 1;

      // If the time step was large enough to move forward more
      // then one data file we need to load both current files
      // to stay accurate in interpolation
      if (std::abs(current_file_number - old_file_number) >= 1)
        try
          {
            if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
              surface_lookup->load_file (create_surface_filename (current_file_number),
                                         this->get_pcout());

            if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
              {
                this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                      << create_filename (current_file_number) << "." << std::endl << std::endl;
                lookup->load_file (create_filename (current_file_number));
              }

          }
        catch (...)
          // If loading current_time_step failed, end time dependent part with old_file_number.
          {
            try
              {
                end_time_dependence (old_file_number);
                return;
              }
            catch (...)
              {
                // If loading the old file fails (e.g. there was no old file), cancel the model run.
                // We might get here, if the model time step is so large that step t is before the
                // whole boundary condition while step t+1 is already behind all files in time.
                AssertThrow (false,
                             ExcMessage (
                               "Loading new and old data file did not succeed. "
                               "Maybe the time step was so large we jumped over all files "
                               "or the files were removed during the model run. "
                               "Another possible way here is to restart a model with "
                               "previously time-dependent boundary condition after the "
                               "last file was already read. Aspect has no way to find the "
                               "last readable file from the current model time. Please "
                               "prescribe the last data file manually in such a case. "
                               "Cancelling calculation."));
              }
          }

      // Now load the data file. This part is the main purpose of this function.
      try
        {
          if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
            surface_lookup->load_file (create_surface_filename (next_file_number),
                                       this->get_pcout());

          if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
            {
              this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                    << create_filename (next_file_number) << "." << std::endl << std::endl;
              lookup->load_file (create_filename (next_file_number));
            }
        }

      // If loading current_time_step + 1 failed, end time dependent part with current_time_step.
      // We do not need to check for success here, because current_file_number was guaranteed to be
      // at least tried to be loaded before, and if it fails, it should have done before (except from
      // hard drive errors, in which case the exception is the right thing to be thrown).

      catch (...)
        {
          end_time_dependence (current_file_number);
        }
    }


    template <int dim>
    void
    PlumeBox<dim>::end_time_dependence (const int file_number)
    {
      // Next data file not found --> Constant velocities
      // by simply loading the old file twice
      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "bottom")
        {
           double time = this->get_time();

          // This happens if this function is called from initialize()
          // In this case we pretend to be at the end of time to
          // enforce the time_weight to use the last file available
          if (std::isnan(time))
            time = 0.0;

          surface_lookup->update(time - first_data_file_model_time);
          surface_lookup->load_file (create_surface_filename (file_number),
                                     this->get_pcout());
        }

      if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) != "top")
        {
          this->get_pcout() << std::endl << "   Loading Ascii data boundary file "
                << create_filename (file_number) << "." << std::endl << std::endl;
          lookup->load_file (create_filename (file_number));
        }

      // no longer consider the problem time dependent from here on out
      // this cancels all attempts to read files at the next time steps
      time_dependent = false;
      // this cancels the time interpolation in lookup
      time_weight = 1.0;
      // Give warning if first processor
      this->get_pcout() << std::endl
                        << "   Loading new data file did not succeed." << std::endl
                        << "   Assuming constant boundary conditions for rest of model run."
                        << std::endl << std::endl;
    }


    template <int dim>
    Tensor<1,dim>
    PlumeBox<dim>::
    boundary_velocity (const Point<dim> &position) const
    {
      if (this->get_time() - first_data_file_model_time >= 0.0)
        {
          Tensor<1,dim> velocity;
          // If at the top, only use plate velocity
          if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) == "top")
            velocity = surface_lookup->surface_velocity(position,time_weight);
          // If at the bottom, add the plume contribution
          else if (this->get_geometry_model().translate_id_to_symbol_name(boundary_id) == "bottom")
            {
              for (unsigned int i = 0; i < dim; i++)
                velocity[i] = lookup->get_data(position,i,time_weight);
              velocity[dim-1] += V0 * std::exp(-std::pow((position-plume_position).norm()/R0,2));
            }
          // At the sides interpolate between side and top velocity
          else
            {
              const double depth = this->get_geometry_model().depth(position);
              const double depth_weight = 0.5*(1.0 - std::tanh((depth - lithosphere_thickness)
                                                               / depth_interpolation_width));

              for (unsigned int i = 0; i < dim; i++)
                velocity[i] = (1 - depth_weight) * lookup->get_data(position,i,time_weight);

              velocity += depth_weight * surface_lookup->surface_velocity(position,time_weight);
            }

          return velocity;
        }
      else
        return Tensor<1,dim> ();
    }


    template <int dim>
    void
    PlumeBox<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Boundary velocity model");
      {
        prm.enter_subsection ("Ascii data model");
        {
          prm.declare_entry ("Data directory",
                             "$ASPECT_SOURCE_DIR/data/velocity-boundary-conditions/ascii-data/test/",
                             Patterns::DirectoryName (),
                             "The name of a directory that contains the model data. This path "
                             "may either be absolute (if starting with a '/') or relative to "
                             "the current directory. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the 'data/' subdirectory of ASPECT. ");
          prm.declare_entry ("Data file name",
                             "box_2d_%s.%d.csv",
                             Patterns::Anything (),
                             "The file name of the material data. Provide file in format: "
                             "(Velocity file name).\\%s%d where \\\\%s is a string specifying "
                             "the boundary of the model according to the names of the boundary "
                             "indicators (of a box or a spherical shell).%d is any sprintf integer "
                             "qualifier, specifying the format of the current file number. ");
          prm.declare_entry ("Data file time step", "1e6",
                             Patterns::Double (0),
                             "Time step between following velocity files. "
                             "Depending on the setting of the global 'Use years in output instead of seconds' flag "
                             "in the input file, this number is either interpreted as seconds or as years. "
                             "The default is one million, i.e., either one million seconds or one million years.");
          prm.declare_entry ("Number of x grid points", "0",
                             Patterns::Double (0),
                             "Number of grid points in x direction. You need to either set this variable in the "
                             "parameter file or provide a '# POINTS: x [y]' line in your first data file. You "
                             "can also set both or include the line in every data file, but in this case ASPECT "
                             "crashes if there are conflicts between the numbers. Dimensions that are not used "
                             "in the model or for this boundary plugin are ignored.");
          prm.declare_entry ("Number of y grid points", "0",
                             Patterns::Double (0),
                             "Number of grid points in y direction.");
          prm.declare_entry ("Number of z grid points", "0",
                             Patterns::Double (0),
                             "Number of grid points in z direction.");
          prm.declare_entry ("First data file model time", "0",
                             Patterns::Double (0),
                             "Time from which on the velocity file with number 'First velocity "
                             "file number' is used as boundary condition. Previous to this "
                             "time, a no-slip boundary condition is assumed. Depending on the setting "
                             "of the global 'Use years in output instead of seconds' flag "
                             "in the input file, this number is either interpreted as seconds or as years.");
          prm.declare_entry ("First data file number", "0",
                             Patterns::Integer (),
                             "Number of the first velocity file to be loaded when the model time "
                             "is larger than 'First velocity file model time'.");
          prm.declare_entry ("Decreasing file order", "false",
                             Patterns::Bool (),
                             "In some cases the boundary files are not numbered in increasing "
                             "but in decreasing order (e.g. 'Ma BP'). If this flag is set to "
                             "'True' the plugin will first load the file with the number "
                             "'First velocity file number' and decrease the file number during "
                             "the model run.");
          prm.declare_entry ("Scale factor", "1",
                             Patterns::Double (0),
                             "Scalar factor, which is applied to the boundary velocity. "
                             "You might want to use this to scale the velocities to a "
                             "reference model (e.g. with free-slip boundary) or another "
                             "plate reconstruction. Another way to use this factor is to "
                             "convert units of the input files. The unit is assumed to be"
                             "m/s or m/yr depending on the 'Use years in output instead of "
                             "seconds' flag. If you provide velocities in cm/yr set this "
                             "factor to 0.01.");
        }
        prm.leave_subsection();

        prm.enter_subsection ("Box plates model");
        {
          prm.declare_entry ("Data directory",
                             "$ASPECT_SOURCE_DIR/data/velocity-boundary-conditions/tristan_plume/",
                             Patterns::DirectoryName (),
                             "The name of a directory that contains the model data. This path "
                             "may either be absolute (if starting with a '/') or relative to "
                             "the current directory. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the 'data/' subdirectory of ASPECT. ");
          prm.declare_entry ("Id file names", "plates_%d",
                             Patterns::Anything (),
                             "The file name of the id data. Provide file in format: "
                             "(Velocity file name).\\%d.gpml where \\%d is any sprintf integer "
                             "qualifier, specifying the format of the current file number.");
          prm.declare_entry ("Velocity file name", "velocities",
                             Patterns::Anything (),
                             "The file name of the material data. Provide file in format: "
                             "(Velocity file name).\\%d.gpml where \\%d is any sprintf integer "
                             "qualifier, specifying the format of the current file number.");
          prm.declare_entry ("X grid spacing", "20e3",
                             Patterns::Double (0),
                             "Distance between two grid points in x direction.");
          prm.declare_entry ("Y grid spacing", "20e3",
                             Patterns::Double (0),
                             "Distance between two grid points in y direction.");
          prm.declare_entry ("Interpolation width", "0",
                             Patterns::Double (0),
                             "Determines the width of the velocity interpolation zone around the current point. "
                             "Currently equals the distance between evaluation point and velocity data point that "
                             "is still included in the interpolation. The weighting of the points currently only accounts "
                             "for the surface area a single data point is covering ('moving window' interpolation without "
                             "distance weighting).");
          prm.declare_entry ("Lithosphere thickness", "2e5",
                             Patterns::Double (0),
                             "The velocity at the sides is interpolated between a plate velocity "
                             "at the surface and ascii data velocities below the lithosphere. "
                             "This value is the depth of the transition between surface and side "
                             "velocities.");
          prm.declare_entry ("Depth interpolation width", "5e4",
                             Patterns::Double (0),
                             "The width of the interpolation zone described for 'lithosphere "
                             "thickness'.");
          prm.declare_entry ("Time scale factor", "1e6",
                             Patterns::Double (0),
                             "Determines the factor applied to the times in the velocity file. The unit is assumed to be"
                             "s or yr depending on the 'Use years in output instead of "
                             "seconds' flag. If you provide time in Ma set this "
                             "factor to 1e6.");
          prm.declare_entry ("Scale factor", "1",
                             Patterns::Double (0),
                             "Scalar factor, which is applied to the boundary velocity. "
                             "You might want to use this to scale the velocities to a "
                             "reference model (e.g. with free-slip boundary) or another "
                             "plate reconstruction.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Plume");
        {
          prm.declare_entry ("Data directory",
                             "$ASPECT_SOURCE_DIR/data/boundary-temperature/plume/",
                             Patterns::DirectoryName (),
                             "The name of a directory that contains the model data. This path "
                             "may either be absolute (if starting with a '/') or relative to "
                             "the current directory. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the 'data/' subdirectory of ASPECT. ");
          prm.declare_entry ("Plume position file name", "Tristan.sur",
                             Patterns::Anything (),
                             "The file name of the plume position data.");
          prm.declare_entry ("Inflow velocity", "0",
                             Patterns::Double (),
                             "Magnitude of the velocity inflow. Units: K.");
          prm.declare_entry ("Radius", "0",
                             Patterns::Double (),
                             "Radius of the anomaly. Units: m.");
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    PlumeBox<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Boundary velocity model");
      {
        prm.enter_subsection("Ascii data model");
        {
          // Get the path to the data files. If it contains a reference
          // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
          // as a #define
          data_directory    = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = data_directory.find (subst_text),  position!=std::string::npos)
              data_directory.replace (data_directory.begin()+position,
                                      data_directory.begin()+position+subst_text.size(),
                                      ASPECT_SOURCE_DIR);
          }

          data_file_name    = prm.get ("Data file name");

          scale_factor      = prm.get_double ("Scale factor");
          data_points[0]    = prm.get_double ("Number of x grid points");
          data_points[1]    = prm.get_double ("Number of y grid points");
          data_points[2]    = prm.get_double ("Number of z grid points");

          data_file_time_step             = prm.get_double ("Data file time step");
          first_data_file_model_time      = prm.get_double ("First data file model time");
          first_data_file_number          = prm.get_double ("First data file number");
          decreasing_file_order           = prm.get_bool   ("Decreasing file order");

          if (this->convert_output_to_years() == true)
            {
              data_file_time_step        *= year_in_seconds;
              first_data_file_model_time *= year_in_seconds;
              scale_factor               /= year_in_seconds;
            }
        }
        prm.leave_subsection();

        prm.enter_subsection("Box plates model");
        {
          // Get the path to the data files. If it contains a reference
          // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
          // as a #define
          surface_data_directory        = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = surface_data_directory.find (subst_text),  position!=std::string::npos)
              surface_data_directory.replace (surface_data_directory.begin()+position,
                                      surface_data_directory.begin()+position+subst_text.size(),
                                      ASPECT_SOURCE_DIR);
          }

          surface_velocity_file_name    = prm.get ("Velocity file name");
          surface_id_file_names         = prm.get ("Id file names");

          interpolation_width   = prm.get_double ("Interpolation width");
          lithosphere_thickness = prm.get_double ("Lithosphere thickness");
          depth_interpolation_width     = prm.get_double ("Depth interpolation width");
          x_step                = prm.get_double ("X grid spacing");
          y_step                = prm.get_double ("Y grid spacing");
          surface_time_scale_factor     = prm.get_double ("Time scale factor");
          surface_scale_factor  = prm.get_double ("Scale factor");

          if (this->convert_output_to_years() == true)
            {
              surface_time_scale_factor *= year_in_seconds;
              surface_scale_factor      /= year_in_seconds;
            }
        }
        prm.leave_subsection();

      }
      prm.leave_subsection();

      prm.enter_subsection("Plume");
      {
        // Get the path to the data files. If it contains a reference
         // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
         // as a #define
         plume_data_directory        = prm.get ("Data directory");
         {
           const std::string      subst_text = "$ASPECT_SOURCE_DIR";
           std::string::size_type position;
           while (position = plume_data_directory.find (subst_text),  position!=std::string::npos)
             plume_data_directory.replace (plume_data_directory.begin()+position,
                                     plume_data_directory.begin()+position+subst_text.size(),
                                     ASPECT_SOURCE_DIR);
         }

         plume_file_name    = prm.get ("Plume position file name");
         V0 = prm.get_double ("Inflow velocity");
         R0 = prm.get_double ("Radius");

         if (this->convert_output_to_years() == true)
           {
             V0 /= year_in_seconds;
           }
      }
      prm.leave_subsection ();
    }

  }
}

// explicit instantiations
namespace aspect
{
  namespace VelocityBoundaryConditions
  {
    ASPECT_REGISTER_VELOCITY_BOUNDARY_CONDITIONS(PlumeBox,
                                                 "plume box",
                                                 "This is a velocity plugin that is a combination "
                                                 "of the Ascii data, Box plates and Plume plugins. "
                                                 "It is specifically designed for modelling a box "
                                                 "of the mantle with prescribed velocities at every "
                                                 "boundary with a plume influx at the bottom, a plate "
                                                 "velocity at the top, side and bottom velocities from "
                                                 "a different geodynamic model and interpolation "
                                                 "between these constraints.")
  }
}
