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


#include <aspect/global.h>
#include <aspect/velocity_boundary_conditions/gplates.h>
#include <aspect/geometry_model/spherical_shell.h>
#include <aspect/utilities.h>

#include <deal.II/base/utilities.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <aspect/geometry_model/box.h>
#include <aspect/geometry_model/spherical_shell.h>


namespace aspect
{
  namespace VelocityBoundaryConditions
  {
    namespace internal
    {
      template <int dim>
      Mapping<dim>::Mapping()
      {}

      template <int dim>
      Point<3>
      Mapping<dim>::map_box_to_sphere_coordinates (Tensor<1,3> &bposition) const
      {
        Point<3> lambert_coord;

        lambert_coord[0] = sqrt(1 - (((bposition[0] * bposition[0]) + (bposition[1] * bposition[1])) / 4)) * bposition[0];  // x
        lambert_coord[1] = sqrt(1 - (((bposition[0] * bposition[0]) + (bposition[1] * bposition[1])) / 4)) * bposition[1];  // y
        lambert_coord[2] =    (-1)+ (((bposition[0] * bposition[0]) + (bposition[1] * bposition[1])) / 2);                  // z

        return lambert_coord;
       }

      template <int dim>
      Point<2>
      Mapping<dim>::map_sphere_to_box_coordinates (Point<3> &sposition) const
      {
        Point<2> lambert_coord;

        lambert_coord[0] = sqrt(2/(1-sposition[2]))*sposition[0];  // X
        lambert_coord[1] = sqrt(2/(1-sposition[2]))*sposition[1];  // Y

        return lambert_coord;
      }

      template <int dim>
      Tensor<1,2>
      Mapping<dim>::map_box_to_sphere_velocities(Tensor<1,2> &bvelocities) const
      {
       Tensor<1,2> svelocities;

       // TODO: actual mapping procedure needs to be implemented!!!
       svelocities[0] = bvelocities[0];
       svelocities[1] = bvelocities[1];

       return svelocities;
      }

      template <int dim>
      Tensor<1,dim>
      Mapping<dim>::map_sphere_to_box_velocities(Tensor<1,dim> &svelocities) const
      {
       Tensor<1,dim> bvelocities;

       // TODO: actual mapping procedure needs to be implemented!!!

       return bvelocities;
      }




      template <int dim>
      GPlatesLookup<dim>::GPlatesLookup(const Tensor<1,2> &surface_point_one,
                                        const Tensor<1,2> &surface_point_two,
                                        Mapping<dim> &map)
                                        :
                                        velocities(2)
      {
        mapping.reset(&map);

        GPlatesLookup(surface_point_one, surface_point_two);

      }

      template <int dim>
      GPlatesLookup<dim>::GPlatesLookup(const Tensor<1,2> &surface_point_one,
                                        const Tensor<1,2> &surface_point_two)
                                        :
                                        velocities(2)
      {
        // get the Cartesian coordinates of the points the 2D model will lie in
        // this computation is done also for 3D since it is not expensive and the
        // template dim is currently not used here. Could be changed.
        const Tensor<1,3> point_one = cartesian_surface_coordinates(convert_tensor<2,3>(surface_point_one));
        const Tensor<1,3> point_two = cartesian_surface_coordinates(convert_tensor<2,3>(surface_point_two));

        // Set up the normal vector of an unrotated 2D spherical shell
        // that by default lies in the x-y plane.
        const double normal[3] = {0.0,0.0,1.0};
        const Tensor<1,3> unrotated_normal_vector (normal);

        // Compute the normal vector of the plane that contains
        // the origin and the two user-specified points
        Tensor<1,3> rotated_normal_vector = cross_product_3d(point_one,point_two);

        rotated_normal_vector /= rotated_normal_vector.norm();

        if ((rotated_normal_vector - unrotated_normal_vector).norm() > 1e-3)
          {
            // Calculate the crossing line of the two normals,
            // which will be the rotation axis to transform the one
            // normal into the other
            Tensor<1,3> rotation_axis = cross_product_3d(unrotated_normal_vector,rotated_normal_vector);
            rotation_axis /= rotation_axis.norm();

            // Calculate the rotation angle from the inner product rule
            const double rotation_angle = std::acos(rotated_normal_vector*unrotated_normal_vector);

            rotation_matrix = rotation_matrix_from_axis(rotation_axis,rotation_angle);

            // Now apply the rotation that will project point_one onto the known point
            // (0,1,0).
            const Tensor<1,3> rotated_point_one = transpose(rotation_matrix) * point_one;

            const double point_one_coords[3] = {0.0,1.0,0.0};
            const Tensor<1,3> final_point_one (point_one_coords);

            const double second_rotation_angle = std::acos(rotated_point_one*final_point_one);
            Tensor<1,3> second_rotation_axis = cross_product_3d(final_point_one,rotated_point_one);
            second_rotation_axis /= second_rotation_axis.norm();

            const Tensor<2,3> second_rotation_matrix = rotation_matrix_from_axis(second_rotation_axis,second_rotation_angle);

            // The final rotation used for the model will be the combined
            // rotation of the two operation above. This is achieved by a
            // matrix multiplication of the rotation matrices.
            // This concatenation of rotations is the reason for using a
            // rotation matrix instead of a combined rotation_axis + angle
            rotation_matrix = rotation_matrix * second_rotation_matrix;
          }
        else
          {
            rotation_matrix[0][0] = 1.0;
            rotation_matrix[1][1] = 1.0;
            rotation_matrix[2][2] = 1.0;
          }
      }

      template <int dim>
      void GPlatesLookup<dim>::screen_output(const Tensor<1,2> &surface_point_one,
                                        const Tensor<1,2> &surface_point_two,
                                        const ConditionalOStream &pcout) const
      {
        const Tensor<1,3> point_one = cartesian_surface_coordinates(convert_tensor<2,3>(surface_point_one));
        const Tensor<1,3> point_two = cartesian_surface_coordinates(convert_tensor<2,3>(surface_point_two));

        std::ostringstream output;

        output << std::setprecision (3) << std::setw(3) << std::fixed << std::endl
               << "   Setting up GPlates boundary velocity plugin."  << std::endl
               << std::endl;
        if (dim == 2)
          {
            Tensor<1,3> rotation_axis;
            const double rotation_angle = rotation_axis_from_matrix(rotation_axis,rotation_matrix);

            std_cxx11::array<double,3> angles = angles_from_matrix(rotation_matrix);
            std_cxx11::array<double,3> back_angles = angles_from_matrix(transpose(rotation_matrix));

            output << "   Input point 1 spherical coordinates: " << surface_point_one  << std::endl
                   << "   Input point 1 normalized cartesian coordinates: " << point_one  << std::endl
                   << "   Input point 1 rotated model coordinates: " << transpose(rotation_matrix) * point_one  << std::endl
                   << "   Input point 2 spherical coordinates: " << surface_point_two  << std::endl
                   << "   Input point 2 normalized cartesian coordinates: " << point_two  << std::endl
                   << "   Input point 2 rotated model coordinates: " << transpose(rotation_matrix) * point_two << std::endl
                   << std::endl <<  std::setprecision(2)
                   << "   Model will be rotated by " << -rotation_angle*180/numbers::PI
                   << " degrees around axis " << rotation_axis << std::endl
                   << "   The ParaView rotation angles are: " << angles[0] << " " << angles [1] << " " << angles[2] << std::endl
                   << "   The inverse ParaView rotation angles are: " << back_angles[0] << " " << back_angles [1] << " " << back_angles[2]

                   << std::endl;
          }

        pcout << output.str();
      }

      template <int dim>
      void
      GPlatesLookup<dim>::load_file(const std::string &filename, const ConditionalOStream &pcout)
      {
        pcout << std::endl << "   Loading GPlates boundary velocity file "
              << filename << "." << std::endl << std::endl;

        using boost::property_tree::ptree;
        ptree pt;

        // Check whether file exists, we do not want to throw
        // an exception in case it does not, because it could be by purpose
        // (i.e. the end of the boundary condition is reached)
        AssertThrow (Utilities::fexists(filename),
                     ExcMessage (std::string("GPlates file <")
                                 +
                                 filename
                                 +
                                 "> not found!"));

        // populate tree structure pt
        read_xml(filename, pt);

        const unsigned int n_points = pt.get_child("gpml:FeatureCollection.gml:featureMember.gpml:VelocityField.gml:domainSet.gml:MultiPoint").size();

        // These formulas look magic, but they are the proper solution to the equation:
        // n_points = n_theta * n_phi with n_phi = 2 * (n_theta - 1)
        // From the XML information we only know n_points, but need n_theta
        // and n_phi to properly size the arrays and get the grip point positions
        const double dn_theta = 0.5 + std::sqrt(0.25 + n_points/2);
        const unsigned int n_theta = static_cast<unsigned int> (dn_theta);
        const unsigned int n_phi = static_cast<unsigned int> (2 * (dn_theta - 1));

        AssertThrow(dn_theta - n_theta <= 1e-5,
                    ExcMessage("The velocity file has a grid structure that is not readable. Please refer to the manual for a proper grid structure."));

        delta_theta =   numbers::PI / (n_theta-1);
        delta_phi   = 2*numbers::PI / n_phi;

        /**
         * Tables which contain the velocities
         */
        std::vector<Table<2,double> > velocity_values(2,Table<2,double>(n_theta,n_phi));

        std::string velos = pt.get<std::string>("gpml:FeatureCollection.gml:featureMember.gpml:VelocityField.gml:rangeSet.gml:DataBlock.gml:tupleList");
        std::stringstream in(velos, std::ios::in);
        AssertThrow (in,
                     ExcMessage (std::string("Couldn't find velocities. Is file native gpml format for velocities?")));

        // The lat-lon mesh has changed its starting longitude in gplates1.4
        // correct for this while reading in the velocity data
        unsigned int longitude_correction = 0;
        if (gplates_1_4_or_higher(pt))
          longitude_correction = n_phi/2;

        unsigned int i = 0;
        char sep;
        Tensor<1,2> spherical_velocities;

        while (in >> spherical_velocities[0]>> sep >> spherical_velocities[1])
          {
            const double cmyr_si = 0.01/year_in_seconds;

            const unsigned int idx_theta = i / n_phi;
            const unsigned int idx_phi = (i + longitude_correction) % n_phi;

            velocity_values[0][idx_theta][idx_phi]= spherical_velocities[0] * cmyr_si;
            velocity_values[1][idx_theta][idx_phi]= spherical_velocities[1] * cmyr_si;

            i++;
          }


            // number of intervals in the direction of theta and phi
            std_cxx11::array<unsigned int,2> table_intervals;
            table_intervals[0] = n_theta - 1;
            table_intervals[1] = n_phi - 1;


            // min and max extent of the grid in the direction of theta and phi (whole spheres in GPlates)
            // polar angle theta: from 0° to 180°(PI)
            grid_extent[0].first = 0;
            grid_extent[0].second = numbers::PI;
            // azimuthal angle phi: from 0° to 360°(2*PI)
            grid_extent[1].first = 0;
            grid_extent[1].second = 2 * numbers::PI;


            for (unsigned int i = 0; i < 2; i++)
              {
                if (velocities[i])
                  delete velocities[i];
                velocities[i] = new Functions::InterpolatedUniformGridData<2> (grid_extent,
                                                                               table_intervals,
                                                                               velocity_values[i]);
          }

        AssertThrow(i == n_points,
                    ExcMessage (std::string("Number of read in points does not match number of points in file. File corrupted?")));
      }

      template <int dim>
      Tensor<1,dim>
      GPlatesLookup<dim>::surface_velocity(const Point<dim> &position,
                                           const double time_weight) const
      {
        Tensor<1,3> internal_position;
        if (dim == 2)
          internal_position = rotation_matrix * convert_tensor<dim,3>(position);
        else
          internal_position = convert_tensor<dim,3>(position);

        // for box geometry, cartesian coordinates of the box (plane) are mapped into cartesian coordinates on a sphere
        Point<3> internal_position_mapped;
        if (mapping != 0)
          internal_position_mapped = mapping -> map_box_to_sphere_coordinates (internal_position);
        else
          internal_position_mapped = internal_position;

        //transform internal_position in spherical coordinates
        const std_cxx11::array<double,3> internal_position_in_spher_array =
            ::aspect::Utilities::spherical_coordinates(Point<3>(internal_position_mapped));

        //remove the radius (first entry of internal_position_in_spher_array)
        Point<2> internal_position_in_spher_rad;

        for (unsigned int i = 1; i < 3; i++)
          {
            internal_position_in_spher_rad[i-1] = internal_position_in_spher_array[3-i];
          }

        // Main work, interpolate velocity at this point
        Tensor<1,2> interpolated_velocity;

        for (unsigned int i = 0; i < 2; i++)
          {
            interpolated_velocity[i] = velocities[i]->value(internal_position_in_spher_rad);
          }


        // for box geometry, velocities of the box (plane) are mapped into velocities on a sphere
        Tensor<1,2> interpolated_velocity_mapped;
        if (mapping != 0)
          interpolated_velocity_mapped = mapping -> map_box_to_sphere_velocities (interpolated_velocity);
        else
          interpolated_velocity_mapped = interpolated_velocity;

        //transform interpolated_velocity in cartesian coordinates
        Tensor<1,3> interpolated_velocity_in_cart;
        interpolated_velocity_in_cart = sphere_to_cart_velocity(interpolated_velocity_mapped,internal_position_in_spher_array);


        Tensor<1,dim> output_boundary_velocity;

        if (dim == 2)
          // convert_tensor conveniently also handles the projection to the 2D plane by
          // omitting the z-component of velocity (since the 2D model lies in the x-y plane).
          output_boundary_velocity = convert_tensor<3,dim>(transpose(rotation_matrix) * interpolated_velocity_in_cart);
        else
          output_boundary_velocity = convert_tensor<3,dim>(interpolated_velocity_in_cart);

        return output_boundary_velocity;
      }


      template <int dim>
      Tensor<1,3>
      GPlatesLookup<dim>::rotate_grid_velocity(const Tensor<1,3> &data_position,
                                          const Tensor<1,3> &point_position,
                                          const Tensor<1,3> &data_velocity) const
      {

        if ((point_position-data_position).norm()/point_position.norm() < 1e-7)
          return data_velocity;
        else
          {
            Tensor<1,3> local_rotation_axis = cross_product_3d(data_position,point_position);
            local_rotation_axis /= local_rotation_axis.norm();

            // Calculate the rotation angle from the inner product rule
            const double local_rotation_angle = std::acos(data_position*point_position);

            const Tensor<1,3> point_velocity = rotate_around_axis(data_velocity,local_rotation_axis,local_rotation_angle);

            return point_velocity;
          }
      }


      template <int dim>
      Tensor<1,3>
      GPlatesLookup<dim>::get_grid_point_position(const unsigned int theta_index, const unsigned int phi_index, const bool cartesian) const
      {
        Tensor<1,3> spherical_position;
        spherical_position[0] = theta_index * delta_theta;
        spherical_position[1] = phi_index * delta_phi;
        spherical_position[2] = 1.0;

        if (cartesian)
          return cartesian_surface_coordinates(spherical_position);
        else
          return spherical_position;
      }

      template <int dim>
      double
      GPlatesLookup<dim>::arc_distance(const Tensor<1,3> position_1, const Tensor<1,3> position_2) const
      {
        const double cartesian_distance = (position_1-position_2).norm();

        Assert((position_1.norm() - position_2.norm())/position_1.norm() < 1e-14,
               ExcMessage("Error in velocity boundary module interpolation. "
                          "Radius of different surface points is not equal."));

        const double average_radius = (position_1.norm() + position_2.norm())/2;
        const double arc_distance =  average_radius * 2.0 * std::asin(cartesian_distance/(2*average_radius));
        return arc_distance;
      }


      template <int dim>
      Tensor<1,3>
      GPlatesLookup<dim>::rotate_around_axis (const Tensor<1,3> &position, const Tensor<1,3> &rotation_axis, const double angle) const
      {
        Tensor<1,3> cross = cross_product_3d(rotation_axis,position);
        const Tensor<1,3> newpos = (1-std::cos(angle)) * rotation_axis*(rotation_axis*position) +
                                   std::cos(angle) * position + std::sin(angle) * cross;
        return newpos;
      }

      template <int dim>
      Tensor<1,3>
      GPlatesLookup<dim>::cartesian_surface_coordinates(const Tensor<1,3> &sposition) const
      {
        Tensor<1,3> ccoord;

        ccoord[0] = std::sin(sposition[0]) * std::cos(sposition[1]); // X
        ccoord[1] = std::sin(sposition[0]) * std::sin(sposition[1]); // Y
        ccoord[2] = std::cos(sposition[0]); // Z
        return ccoord;
      }

      template <int dim>
      Tensor<1,3>
      GPlatesLookup<dim>::sphere_to_cart_velocity(const Tensor<1,2> &s_velocities, const std_cxx11::array<double,3> &s_position) const
      {
        Tensor<1,3> velocity;

        velocity[0] = -1.0 * s_velocities[1] * std::sin(s_position[1]) + s_velocities[0]*std::cos(s_position[2])*std::cos(s_position[1]);
        velocity[1] = s_velocities[1]*std::cos(s_position[1])+s_velocities[0]*std::cos(s_position[2])*std::sin(s_position[1]);
        velocity[2] = -1.0*s_velocities[0]*std::sin(s_position[2]);
        return velocity;
      }

      template <int dim>
      Tensor<2,3>
      GPlatesLookup<dim>::rotation_matrix_from_axis (const Tensor<1,3> &rotation_axis,
                                                const double rotation_angle) const
      {
        Tensor<2,3> rotation_matrix;
        rotation_matrix[0][0] = (1-std::cos(rotation_angle)) * rotation_axis[0]*rotation_axis[0] + std::cos(rotation_angle);
        rotation_matrix[0][1] = (1-std::cos(rotation_angle)) * rotation_axis[0]*rotation_axis[1] - rotation_axis[2] * std::sin(rotation_angle);
        rotation_matrix[0][2] = (1-std::cos(rotation_angle)) * rotation_axis[0]*rotation_axis[2] + rotation_axis[1] * std::sin(rotation_angle);
        rotation_matrix[1][0] = (1-std::cos(rotation_angle)) * rotation_axis[1]*rotation_axis[0] + rotation_axis[2] * std::sin(rotation_angle);
        rotation_matrix[1][1] = (1-std::cos(rotation_angle)) * rotation_axis[1]*rotation_axis[1] + std::cos(rotation_angle);
        rotation_matrix[1][2] = (1-std::cos(rotation_angle)) * rotation_axis[1]*rotation_axis[2] - rotation_axis[0] * std::sin(rotation_angle);
        rotation_matrix[2][0] = (1-std::cos(rotation_angle)) * rotation_axis[2]*rotation_axis[0] - rotation_axis[1] * std::sin(rotation_angle);
        rotation_matrix[2][1] = (1-std::cos(rotation_angle)) * rotation_axis[2]*rotation_axis[1] + rotation_axis[0] * std::sin(rotation_angle);
        rotation_matrix[2][2] = (1-std::cos(rotation_angle)) * rotation_axis[2]*rotation_axis[2] + std::cos(rotation_angle);
        return rotation_matrix;
      }

      template <int dim>
      double
      GPlatesLookup<dim>::rotation_axis_from_matrix (Tensor<1,3> &rotation_axis,
                                                const Tensor<2,3> &rotation_matrix) const
      {
        double rotation_angle = std::acos(0.5 * (rotation_matrix[0][0] + rotation_matrix[1][1] + rotation_matrix[2][2] - 1));

        if (rotation_angle > std::numeric_limits<double>::min())
          {
            rotation_axis[0] = (rotation_matrix[2][1] - rotation_matrix[1][2]) / (2*std::sin(rotation_angle));
            rotation_axis[1] = (rotation_matrix[0][2] - rotation_matrix[2][0]) / (2*std::sin(rotation_angle));
            rotation_axis[2] = (rotation_matrix[1][0] - rotation_matrix[0][1]) / (2*std::sin(rotation_angle));
          }
        else
          {
            rotation_axis[0] = 0.0;
            rotation_axis[1] = 0.0;
            rotation_axis[2] = 1.0;
          }

        return rotation_angle;
      }

      template <int dim>
      std_cxx11::array<double,3>
      GPlatesLookup<dim>::angles_from_matrix(const Tensor<2,3> &rotation_matrix) const
      {
        std_cxx11::array<double,3> orientation;

        /*
         * The following code is part of the VTK project and copied here for
         * compatibility to the paraview rotation formalism. It is protected by
         * the following license:
         *
         *
         * Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
         * All rights reserved.
         *
         * Redistribution and use in source and binary forms, with or without
         * modification, are permitted under certain conditions. See
         * http://www.kitware.com/Copyright.htm for details.

         * This software is distributed WITHOUT ANY WARRANTY; without even
         * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
         * PURPOSE.  See the above copyright notice for more information.
         *
         *
         * The original code in the VTK-6.0.0 source folder is found in:
         * /Common/Transforms/vtkTransform.cxx in the function GetOrientation()
         *
         */

        // first rotate about y axis
        const double x2 = rotation_matrix[2][0];
        const double y2 = rotation_matrix[2][1];
        const double z2 = rotation_matrix[2][2];

        const double x3 = rotation_matrix[1][0];
        const double y3 = rotation_matrix[1][1];
        const double z3 = rotation_matrix[1][2];

        double d1 = sqrt(x2*x2 + z2*z2);

        double cosTheta, sinTheta;
        if (d1 < std::numeric_limits<double>::min())
          {
            cosTheta = 1.0;
            sinTheta = 0.0;
          }
        else
          {
            cosTheta = z2/d1;
            sinTheta = x2/d1;
          }

        double theta = atan2(sinTheta, cosTheta);
        orientation[1] = - theta * 180 / numbers::PI;

        // now rotate about x axis
        double d = sqrt(x2*x2 + y2*y2 + z2*z2);

        double sinPhi, cosPhi;
        if (d < std::numeric_limits<double>::min())
          {
            sinPhi = 0.0;
            cosPhi = 1.0;
          }
        else if (d1 < std::numeric_limits<double>::min())
          {
            sinPhi = y2/d;
            cosPhi = z2/d;
          }
        else
          {
            sinPhi = y2/d;
            cosPhi = (x2*x2 + z2*z2)/(d1*d);
          }

        double phi = atan2(sinPhi, cosPhi);
        orientation[0] = phi * 180 / numbers::PI;

        // finally, rotate about z
        double x3p = x3*cosTheta - z3*sinTheta;
        double y3p = - sinPhi*sinTheta*x3 + cosPhi*y3 - sinPhi*cosTheta*z3;
        double d2 = sqrt(x3p*x3p + y3p*y3p);

        double cosAlpha, sinAlpha;
        if (d2 < std::numeric_limits<double>::min())
          {
            cosAlpha = 1.0;
            sinAlpha = 0.0;
          }
        else
          {
            cosAlpha = y3p/d2;
            sinAlpha = x3p/d2;
          }

        double alpha = atan2(sinAlpha, cosAlpha);
        orientation[2] = alpha * 180 / numbers::PI;
        return orientation;
      }

      template <int dim>
      template <int in, int out>
      Tensor<1,out>
      GPlatesLookup<dim>::convert_tensor (const Tensor<1,in> &old_tensor) const
      {
        Tensor<1,out> new_tensor;
        for (unsigned int i = 0; i < out; i++)
          if (i < in) new_tensor[i] = old_tensor[i];
          else new_tensor[i] = 0.0;

        return new_tensor;
      }


      template <int dim>
      bool
      GPlatesLookup<dim>::gplates_1_4_or_higher(const boost::property_tree::ptree &pt) const
      {
        const std::string gpml_version = pt.get<std::string>("gpml:FeatureCollection.<xmlattr>.gpml:version");
        const std::vector<std::string> string_versions = dealii::Utilities::split_string_list(gpml_version,'.');
        const std::vector<int> int_versions = dealii::Utilities::string_to_int(string_versions);

        const int gplates_1_3_version[3] = {1,6,322};

        for (unsigned int i = 0; i < int_versions.size(); i++)
          {
            if (int_versions[i] > gplates_1_3_version[i])
              return true;
            if (int_versions[i] < gplates_1_3_version[i])
              return false;
          }

        return false;
      }
    }

    template <int dim>
    GPlates<dim>::GPlates ()
      :
      time_relative_to_vel_file_start_time(0.0),
      current_time_step(-2),
      velocity_file_start_time(0.0),
      time_step(0.0),
      time_weight(0.0),
      time_dependent(true),
      point1("0.0,0.0"),
      point2("0.0,0.0"),
      lookup()
    {}


    template <int dim>
    void
    GPlates<dim>::initialize ()
    {
      char sep;

      std::stringstream streampoint(point1);
      streampoint >> pointone[0] >> sep >> pointone[1];

      std::stringstream streampoint2(point2);
      streampoint2 >> pointtwo[0] >> sep >> pointtwo[1];

      if (dim == 2)
        Assert (pointone != pointtwo,
                ExcMessage ("To define a plane for the 2D model the two assigned points "
                            "may not be equal."));

      internal::Mapping<dim> mapping;

      if ((dynamic_cast<const GeometryModel::Box<dim>*> (&this->get_geometry_model())) == 0)
        lookup.reset(new internal::GPlatesLookup<dim>(pointone,pointtwo,mapping));
      else if ((dynamic_cast<const GeometryModel::SphericalShell<dim>*> (&this->get_geometry_model())) == 0)
        lookup.reset(new internal::GPlatesLookup<dim>(pointone,pointtwo));
      else
        AssertThrow (false,ExcMessage ("This gplates plugin can only be used when using "
                                       "a spherical shell or box geometry."));

      const GeometryModel::Interface<dim> &geometry_model =
        this->get_geometry_model();
    }


    template <int dim>
    std::string
    GPlates<dim>::create_filename (const int timestep) const
    {
      std::string templ = data_directory+velocity_file_name;
      const int size = templ.length();
      char *filename = (char *) (malloc ((size + 10) * sizeof(char)));
      snprintf (filename, size + 10, templ.c_str (), timestep);
      std::string str_filename (filename);
      free (filename);
      return str_filename;
    }


    template <int dim>
    void
    GPlates<dim>::update ()
    {
      Interface<dim>::update ();

      time_relative_to_vel_file_start_time = this->get_time() - velocity_file_start_time;

      // If the boundary condition is constant, switch off time_dependence end leave function.
      // This also sets time_weight to 1.0
      if ((create_filename (current_time_step) == create_filename (current_time_step+1)) && time_dependent)
        {
          lookup->screen_output (pointone, pointtwo,this->get_pcout());

          lookup->load_file (create_filename (current_time_step),
                             this->get_pcout());
          end_time_dependence (current_time_step);
          return;
        }

      if (time_dependent && (time_relative_to_vel_file_start_time >= 0.0))
        {
          if (current_time_step < 0)
            lookup->screen_output (pointone, pointtwo,this->get_pcout());

          if (static_cast<int> (time_relative_to_vel_file_start_time / time_step) > current_time_step)
            {
              update_velocity_data();
            }

          time_weight = time_relative_to_vel_file_start_time / time_step - current_time_step;

          Assert ((0 <= time_weight) && (time_weight <= 1),
                  ExcMessage (
                    "Error in set_current_time. Time_weight has to be in [0,1]"));
        }
    }

    template <int dim>
    void
    GPlates<dim>::update_velocity_data ()
    {

      const int old_time_step = current_time_step;
      current_time_step =
        static_cast<unsigned int> (time_relative_to_vel_file_start_time / time_step);
      // Load next velocity file for interpolation
      // If the time step was large enough to move forward more
      // then one velocity file, we need to load both current files
      // to stay accurate in interpolation
      if (current_time_step > old_time_step + 1)
        try
          {
            lookup->load_file (create_filename (current_time_step),
                               this->get_pcout());
          }
        catch (...)
          // If loading current_time_step failed, end time dependent part with old_time_step.
          {
            try
              {
                end_time_dependence (old_time_step);
              }
            catch (...)
              {
                // If loading the old file fails (e.g. there was no old file), cancel the model run.
                // We might get here, if the time step is so large that step t is before the whole boundary condition
                // while step t+1 is already behind all files in time.
                AssertThrow (false,
                             ExcMessage (
                               "Loading new and old velocity file did not succeed. "
                               "Maybe the time step was so large we jumped over all files "
                               "or the files were removed during the model run. "
                               "Another possible way here is to restart a model with "
                               "previously time-dependent boundary condition after the "
                               "last file was already read. Aspect has no way to find the "
                               "last readable file from the current model time. Please "
                               "prescribe the last velocity file manually in such a case. "
                               "Cancelling calculation."));
              }
          }

      // Now load the next velocity file. This part is the main purpose of this function.
      try
        {
          lookup->load_file (create_filename (current_time_step + 1),
                             this->get_pcout());
        }

      // If loading current_time_step + 1 failed, end time dependent part with current_time_step.
      // We do not need to check for success here, because current_time_step was guaranteed to be
      // at least tried to be loaded before, and if it fails, it should have done before (except from
      // hard drive errors, in which case the exception is the right thing to be thrown).

      catch (...)
        {
          end_time_dependence (current_time_step);
        }
    }

    template <int dim>
    void
    GPlates<dim>::end_time_dependence (const int timestep)
    {
      // Next velocity file not found --> Constant velocities
      // by simply loading the old file twice
      lookup->load_file (create_filename (timestep), this->get_pcout());
      // no longer consider the problem time dependent from here on out
      // this cancels all attempts to read files at the next time steps
      time_dependent = false;
      // this cancels the time interpolation in lookup
      time_weight = 1.0;
      // Give warning if first processor
      this->get_pcout() << std::endl
                        << "   Loading new velocity file did not succeed." << std::endl
                        << "   Assuming constant boundary conditions for rest of model run."
                        << std::endl << std::endl;
    }

    template <int dim>
    Tensor<1,dim>
    GPlates<dim>::
    boundary_velocity (const Point<dim> &position) const
    {
      // check depth of position and use velocity or 0 depending on depth < lithosphere_thickness
      if ((time_relative_to_vel_file_start_time >= 0.0) && (this->get_geometry_model().depth(position) <= lithosphere_thickness))
        return scale_factor * lookup->surface_velocity(position,time_weight);
      else
        return Tensor<1,dim> ();
    }


    template <int dim>
    void
    GPlates<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection ("Boundary velocity model");
      {
        prm.enter_subsection ("GPlates model");
        {
          prm.declare_entry ("Data directory",
                             "$ASPECT_SOURCE_DIR/data/velocity-boundary-conditions/gplates/",
                             Patterns::DirectoryName (),
                             "The name of a directory that contains the model data. This path "
                             "may either be absolute (if starting with a '/') or relative to "
                             "the current directory. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the 'data/' subdirectory of ASPECT. ");
          prm.declare_entry ("Velocity file name", "phi.%d",
                             Patterns::Anything (),
                             "The file name of the material data. Provide file in format: "
                             "(Velocity file name).\\%d.gpml where \\%d is any sprintf integer "
                             "qualifier, specifying the format of the current file number.");
          prm.declare_entry ("Time step", "1e6",
                             Patterns::Double (0),
                             "Time step between following velocity files. "
                             "Depending on the setting of the global 'Use years in output instead of seconds' flag "
                             "in the input file, this number is either interpreted as seconds or as years. "
                             "The default is one million, i.e., either one million seconds or one million years.");
          prm.declare_entry ("Velocity file start time", "0.0",
                             Patterns::Double (0),
                             "Time at which the velocity file with number 0 shall be loaded. Previous to this "
                             "time, a no-slip boundary condition is assumed. "
                             "Depending on the setting of the global 'Use years in output instead of seconds' flag "
                             "in the input file, this number is either interpreted as seconds or as years.");
          prm.declare_entry ("Scale factor", "1",
                             Patterns::Double (0),
                             "Scalar factor, which is applied to the boundary velocity. "
                             "You might want to use this to scale the velocities to a "
                             "reference model (e.g. with free-slip boundary) or another "
                             "plate reconstruction.");
          prm.declare_entry ("Point one", "1.570796,0.0",
                             Patterns::Anything (),
                             "Point that determines the plane in which a 2D model lies in. Has to be in the format 'a,b' where a and b are theta (polar angle)  and phi in radians.");
          prm.declare_entry ("Point two", "1.570796,1.570796",
                             Patterns::Anything (),
                             "Point that determines the plane in which a 2D model lies in. Has to be in the format 'a,b' where a and b are theta (polar angle)  and phi in radians.");
          prm.declare_entry ("Lithosphere thickness", "0",
                             Patterns::Double (0),
                             "Determines the depth of the lithosphere, so that the GPlates velocities can be applied there "
                             "as well as at the surface.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }


    template <int dim>
    void
    GPlates<dim>::parse_parameters (ParameterHandler &prm)
    {
      // Query the unit system for time since we may have to convert below.
      // Note that we can't use this->convert_output_to_years() since this
      // requires the SimulatorAccess base object to have been initialized,
      // but this hasn't happened yet when we get into this function.
      const bool
      use_years_instead_of_seconds
        = prm.get_bool ("Use years in output instead of seconds");

      prm.enter_subsection("Boundary velocity model");
      {
        prm.enter_subsection("GPlates model");
        {
          // Get the path to the data files. If it contains a reference
          // to $ASPECT_SOURCE_DIR, replace it by what CMake has given us
          // as a #define
          data_directory        = prm.get ("Data directory");
          {
            const std::string      subst_text = "$ASPECT_SOURCE_DIR";
            std::string::size_type position;
            while (position = data_directory.find (subst_text),  position!=std::string::npos)
              data_directory.replace (data_directory.begin()+position,
                                      data_directory.begin()+position+subst_text.size(),
                                      ASPECT_SOURCE_DIR);
          }

          velocity_file_name    = prm.get ("Velocity file name");
          scale_factor          = prm.get_double ("Scale factor");
          point1                = prm.get ("Point one");
          point2                = prm.get ("Point two");
          lithosphere_thickness = prm.get_double ("Lithosphere thickness");

          time_step             = prm.get_double ("Time step");
          velocity_file_start_time = prm.get_double ("Velocity file start time");
          if (use_years_instead_of_seconds == true)
            {
              time_step                *= year_in_seconds;
              velocity_file_start_time *= year_in_seconds;
            }
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
  namespace VelocityBoundaryConditions
  {
    ASPECT_REGISTER_VELOCITY_BOUNDARY_CONDITIONS(GPlates,
                                                 "gplates",
                                                 "Implementation of a model in which the boundary "
                                                 "velocity is derived from files that are generated "
                                                 "by the GPlates program.")
  }
}
