import astropy.units as u
import numpy as np
from spectral_cube import SpectralCube

from ..utils.logger import catch_warnings, warning_to_log


class CubeHandler:
    """
    A class for handling spectral cubes.
    """

    @catch_warnings
    @staticmethod
    def load_cube(cube_path: str, cube: SpectralCube | None = None):
        """
        Load a spectral cube from a file.

        Parameters
        ----------
        cube_path : str
            The path to the spectral cube file.
        cube : SpectralCube, optional
            The spectral cube to load.

        Returns
        -------
        cube : SpectralCube
            The loaded spectral cube.

        """

        import warnings

        import astropy.wcs
        from astropy import log

        log.setLevel("ERROR")
        astropy.wcs.log.setLevel("ERROR")

        with warning_to_log():
            warnings.filterwarnings("ignore", category=UserWarning)

            # Read in the datacube
            if cube is None:
                cube_ = SpectralCube.read(cube_path)
                cube_ = CubeHandler._standardize_spectral_axis(cube_)
            else:
                cube_ = cube

            # Reverse the cube if the spectral axis is in the wrong order
            if cube_.spectral_axis.diff()[0] < 0:
                cube_ = cube_[::-1]

        return cube_

    @staticmethod
    def _standardize_spectral_axis(cube: SpectralCube):
        """
        Standardize the spectral axis of a SpectralCube.

        This function ensures that the spectral axis is in velocity units
        with a radio velocity convention.

        Parameters
        ----------
        cube : SpectralCube
            The cube to standardize

        Returns
        -------
        cube : SpectralCube
            The cube with the standardized spectral axis

        """

        # Check if spectral axis is already in velocity units
        if cube.spectral_axis.unit.physical_type == "frequency":
            # Convert frequency to velocity
            rest_freq = cube.header.get("RESTFRQ", None)
            if rest_freq is None:
                raise ValueError(
                    "Frequency axis found but no rest frequency found in header"
                )

            cube = cube.with_spectral_unit(
                u.km / u.s, velocity_convention="radio", rest_value=rest_freq * u.Hz
            )

        elif cube.spectral_axis.unit.physical_type == "length":
            # Convert wavelength to velocity
            rest_wav = cube.header.get("RESTWAV", None)
            if rest_wav is None:
                raise ValueError(
                    "Wavelength axis found but no rest wavelength found in header"
                )

            cube = cube.with_spectral_unit(
                u.km / u.s, velocity_convention="radio", rest_value=rest_wav * u.m
            )

        # If already velocity, just ensure radio convention
        elif cube.velocity_convention != "radio":
            cube = cube.with_spectral_unit(u.km / u.s, velocity_convention="radio")

        return cube

    @staticmethod
    def get_x_axis(scouseobject):
        """
        Returns x_axis for spectra

        Parameters
        ----------
        scouseobject : Instance of the scousepy class

        """
        x = np.array(scouseobject.cube.spectral_axis.value)

        trimids = (x >= np.min(scouseobject.vel_range)) & (
            x <= np.max(scouseobject.vel_range)
        )
        if not any(trimids):
            trimids = np.ones(np.shape(x), dtype=bool)
        xtrim = x[trimids]

        return x, xtrim, trimids

    @staticmethod
    def estimate_noise(
        cube: SpectralCube, max_samples: int = 500, fail_limit: int = 50
    ) -> float:
        """
        Estimate the noise of a spectral cube.

        Parameters
        ----------
        cube : SpectralCube
            The cube to estimate the noise of.
        max_samples : int, optional
            The maximum number of samples to use to estimate the noise.
        fail_limit : int, optional
            The maximum number of times the noise estimation can fail before
            the function returns the median of the MAD of the spectrum.

        Returns
        -------
        noise : float
            The estimated noise of the cube.

        """
        import random

        from astropy.stats import mad_std

        from ..processing.noiseestimator import NoiseEstimator

        keep = cube.mask.include().any(axis=0)

        finiteidxs = np.array(np.where(keep), dtype=np.intp)
        flatidxs = [
            np.ravel_multi_index(
                (finiteidxs[0, i], finiteidxs[1, i]), dims=cube.shape[1:]
            )
            for i in range(len(finiteidxs[0, :]))
        ]
        random_indices = random.sample(list(flatidxs), k=len(flatidxs))
        locations = np.array(np.unravel_index(random_indices, cube.shape[1:]))

        if len(locations[0, :]) > max_samples:
            stop = max_samples
        else:
            stop = len(locations[0, :])

        rmsList = []
        stopcount = 0
        specidx = 0
        stuck = 0

        while stopcount < stop:
            _spectrum = np.array(
                cube[:, locations[0, specidx], locations[1, specidx]].to_value(),
                dtype=np.float64,
            )
            noise = NoiseEstimator(cube.spectral_axis.value, _spectrum)
            rmsVal = noise.rms

            if np.isfinite(rmsVal):
                rmsList.append(rmsVal)
                stopcount, specidx = stopcount + 1, specidx + 1
            else:
                rmsVal = mad_std(_spectrum, ignore_nan=True)
                if np.isfinite(rmsVal):
                    rmsList.append(rmsVal)
                    stopcount, specidx = stopcount + 1, specidx + 1
                else:
                    stuck += 1
                    if stuck == fail_limit:
                        break

        return float(np.median(rmsList))
