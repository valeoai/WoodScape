# Calibration

Each image is provided with a dedicated calibration file including intrinsic and extrinsic parameters as well as a descriptive name. 
The name is one of "FV", "MVR", "MVL" or "RV", short for "Front View", "Mirror View Right", "Mirror View Left", "Rear View". 

## Extrinsic

The vehicle coordinate system, which follows the ISO 8855 convention, is anchored to the ground below the midpoint of the rear axle. The X axis points in driving direction, the Y axis points to the left side of the vehicle and the Z axis points up from the ground. 
The camera sensor's coordinate system is based on OpenCV. The X axis points to the right along the horizontal sensor axis, the Y axis points downwards along the vertical sensor axis and the Z axis points in viewing direction along the optical axis to maintain the right-handed system. 
The values of the translation are given in meters and the rotation is given as a quaternion. They describe the coordinate transformation from the camera coordinate system to the vehicle coordinate system. The rotation matrix can be obtained using e.g. `scipy.spatial.transform.Rotation.from_quat`.

## Intrinsic

The intrinsic calibration is given in a calibration model that describes the radial distortion using a 4th order polynomial 
rho(theta) = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4,
where theta is the angle of incidence with respect to the optical axis and rho is the distance between the image center and projected point. The coefficients k1, k2, k3 and k4 are given in the calibration files.
The image width and height as well as the offset (cx, cy) of the principal point are given in pixels.

For completeness, the projection of a 3D point (X, Y, Z) given in camera coordinates to image coordinates (u, v) looks like:

chi = sqrt( X ** 2 + Y ** 2)
theta = arctan2( chi, Z ) = pi / 2 - arctan2( Z, chi )
rho = rho(theta)
u’ = rho * X / chi if chi != 0 else 0
v’ = rho * Y / chi if chi != 0 else 0
u = u’ + cx + width / 2 - 0.5
v = v’ * aspect_ratio + cy + height / 2 - 0.5

The last two lines show the final conversion to image coordinates assuming that the origin of the image coordinate system is located in the upper left corner with the upper left pixel at (0, 0). The projection itself is implemented in `projection.py` which can be found at https://github.com/valeoai/WoodScape/tree/master/scripts/calibration.
