# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from __future__ import division

import numpy as np

from vispy.util import keys

from vispy.util import transforms
from vispy.util.quaternion import Quaternion
from vispy.visuals.transforms import MatrixTransform
from vispy.scene.cameras.perspective import Base3DRotationCamera, PerspectiveCamera


class ArcballCamera(Base3DRotationCamera):
    """3D camera class that orbits around a center point while
    maintaining a view on a center point.

    For this camera, the ``scale_factor`` indicates the zoom level, and
    the ``center`` indicates the position to put at the center of the
    view.

    Parameters
    ----------
    fov : float
        Field of view. Zero (default) means orthographic projection.
    distance : float | None
        The distance of the camera from the rotation point (only makes sense
        if fov > 0). If None (default) the distance is determined from the
        scale_factor and fov.
    translate_speed : float
        Scale factor on translation speed when moving the camera center point.
    **kwargs : dict
        Keyword arguments to pass to `BaseCamera`.

    Notes
    -----
    Interaction:

        * LMB: orbits the view around its center point.
        * RMB or scroll: change scale_factor (i.e. zoom level)
        * SHIFT + LMB: translate the center point
        * SHIFT + RMB: change FOV

    """

    _state_props = Base3DRotationCamera._state_props + ('_quaternion',)

    def __init__(self, fov=45.0, distance=None, translate_speed=1.0, **kwargs):
        super(ArcballCamera, self).__init__(fov=fov, **kwargs)

        # Set camera attributes
        self._quaternion = Quaternion()
        self.distance = distance  # None means auto-distance
        self.translate_speed = translate_speed
        
        
    def viewbox_mouse_event(self, event):
        """
        The viewbox received a mouse event; update transform
        accordingly.

        Parameters
        ----------
        event : instance of Event
            The event.
        """
        if event.handled or not self.interactive:
            return

        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.type == 'mouse_release':
            self._event_value = None  # Reset
        elif event.type == 'mouse_press':
            event.handled = True
        elif event.type == 'mouse_move':
            if event.press_event is None:
                return
            if 1 in event.buttons and 2 in event.buttons:
                return

            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos
            d = p2 - p1

            if 1 in event.buttons and keys.SHIFT not in  modifiers:
                # Rotate
                self._update_rotation(event)

            elif 2 in event.buttons and not modifiers:
                # Zoom
                if self._event_value is None:
                    self._event_value = (self._scale_factor, self._distance)
                zoomy = (1 + self.zoom_factor) ** d[1]

                self.scale_factor = self._event_value[0] * zoomy
                # Modify distance if its given
                if self._distance is not None:
                    self._distance = self._event_value[1] * zoomy
                self.view_changed()

            elif 1 in event.buttons and keys.SHIFT in modifiers:
                # Translate
                norm = np.mean(self._viewbox.size)
                if self._event_value is None or len(self._event_value) == 2:
                    self._event_value = self.center
                dist = (p1 - p2) / norm * self._scale_factor
                dist[1] *= -1
                # Black magic part 1: turn 2D into 3D translations
                dx, dy, dz = self._dist_to_trans(dist)
                # Black magic part 2: take up-vector and flipping into account
                ff = self._flip_factors
                up, forward, right = self._get_dim_vectors()
                dx, dy, dz = right * dx + forward * dy + up * dz
                dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
                c = self._event_value
                self.center = c[0] + dx, c[1] + dy, c[2] + dz

            elif 2 in event.buttons and keys.SHIFT in modifiers:
                # Change fov
                if self._event_value is None:
                    self._event_value = self._fov
                fov = self._event_value - d[1] / 5.0
                self.fov = min(180.0, max(0.0, fov))

    

    def _update_roll(self, rate):
        self._quaternion = Quaternion.create_from_axis_angle(rate[0] * np.pi * 2 * 1.5, 0, 0, 1) * self._quaternion
        

    def _update_rotation(self, event):
        """Update rotation parmeters based on mouse movement"""
        modifiers = event.mouse_event.modifiers
            
        
        p2 = event.mouse_event.pos
        if self._event_value is None:
            self._event_value = p2
        wh = self._viewbox.size
        if keys.META in modifiers:
            self._update_roll((np.array(p2) - np.array(self._event_value)) / wh)
        else:
            self._quaternion = (Quaternion(*_arcball(p2, wh)) *
                                Quaternion(*_arcball(self._event_value, wh)) *
                                self._quaternion)
        self._event_value = p2
        self.view_changed()

    def _get_rotation_tr(self):
        """Return a rotation matrix based on camera parameters"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        return transforms.rotate(180 * rot / np.pi, (x, z, y))

    def _dist_to_trans(self, dist):
        """Convert mouse x, y movement into x, y, z translations"""
        rot, x, y, z = self._quaternion.get_axis_angle()
        tr = MatrixTransform()
        tr.rotate(180 * rot / np.pi, (x, y, z))
        dx, dz, dy = np.dot(tr.matrix[:3, :3],
                            (dist[0], dist[1], 0.)) * self.translate_speed
        return dx, dy, dz

    def _get_dim_vectors(self):
        # Override vectors, camera has no sense of "up"
        return np.eye(3)[::-1]


def _arcball(xy, wh):
    """Convert x,y coordinates to w,x,y,z Quaternion parameters

    Adapted from:

    linalg library

    Copyright (c) 2010-2015, Renaud Blanch <rndblnch at gmail dot com>
    Licence at your convenience:
    GPLv3 or higher <http://www.gnu.org/licenses/gpl.html>
    BSD new <http://opensource.org/licenses/BSD-3-Clause>
    """
    x, y = xy
    w, h = wh
    r = (w + h) / 2.
    x, y = -(2. * x - w) / r, (2. * y - h) / r
    h = np.sqrt(x*x + y*y)
    return (0., x/h, y/h, 0.) if h > 1. else (0., x, y, np.sqrt(1. - h*h))
