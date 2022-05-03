# Licensed under an MIT open source license - see LICENSE

import numpy as np
from astropy.stats import sigma_clip, mad_std, median_absolute_deviation

class getnoise(object):
    """
    Autonomous calculation of the spectral rms

    Notes
    -----
    The method for identifying noise channels is described in Riener+ 2019,
    and has been adapted from the method in https://github.com/mriener/gausspyplus
    credit for the method design goes to Manuel Riener.

    Most of the technical parts of the code below is lifted directly from
    https://github.com/mriener/gausspyplus/blob/master/gausspyplus/utils/noise_estimation.py
    JDH wrapped it up and tinkered a bit.

    """
    def __init__(self, spectral_axis, spectrum, p_limit=0.02, pad_channels=2,
                    n_mad=5.0, remove_broad=True):

        """
        Parameters
        ----------

        spectral_axis : array
            An array of values corresponding to the spectral axis (e.g. velocity)
        spectrum : numpy.ndarray
            Array of the data values of the spectrum
        p_limit : float
            Maximum probability for consecutive positive/negative channels being
            due to chance
        pad_channels : int
            Number of channels by which an interval (low, upp) gets extended on
            both sides, resulting in (low - pad_channels, upp + pad_channels).
            Used for masking of spectral features
        n_mad : float
            Multiple of the median absolute deviation. Used to identify spikes
            in the data which are not noise. All channels below nmad*MAD are
            considered "noise"
        remove_broad : boolean
            Determines method by which the noise channels are identified.
            Default=True. If True, broad components are removed using a simple
            Markov chain to estimate the probability that consecutive +ve or -ve
            channels are due to noise (assuming Gaussian noise). See
            Riener+ 2019 Appendix A for details. If False, a simple method of
            reflected noise is imposed.

        """

        # Set the global quantities
        self.spectral_axis=spectral_axis
        self.spectrum=spectrum
        self.p_limit=p_limit
        self.pad_channels=pad_channels
        self.n_mad=n_mad
        self.n_channels=np.size(spectral_axis)
        self.mask=np.zeros(self.n_channels, dtype='bool')
        self.consecutive_channels=None
        self.ranges=None
        self.max_consecutive_channels=None
        self.noise=None
        self.spectrum_masked=np.copy(self.spectrum)
        self.flag=''

        # fail safe checks for dealing with troublesome spectra
        if np.isfinite(self.spectrum).any():
            if not ((self.spectrum>0.0).all() or (self.spectrum<0.0).all()):
                # So long as not all of the channels are nan, all positive or all
                # negative, we can go ahead and calculate the noise
                self.calculate_noise(remove_broad=remove_broad)
            else:
                # if all channels are either positive or negative then return
                # nan - indicates poor baselining
                self.rms=np.nan
                self.flag='Spectrum is all positive/negative'
        else:
            # if all channels are nan values then return nan
            self.rms=np.nan
            self.flag='Spectrum is all NaNs'

    def calculate_noise(self, remove_broad=True):
        """
        Routine for calculating the noise

        """
        # find peaks within the spectrum
        self.consecutive_channels, self.ranges=self.determine_peaks(self.spectrum)

        # start by masking any nan values
        if np.isnan(self.spectrum).any():
            mask=np.isnan(self.spectrum)
            self.mask+=mask

        self.spectrum_masked[self.mask] = 0

        if remove_broad:
            # start by masking broad features...

            # determine the maximum number of consecutive positive/negative
            # channels being due to chance
            self.max_consecutive_channels=self.get_max_consecutive_channels(self.n_channels,self.p_limit)
            # create the mask
            mask = self.mask_broadpeaks(self.n_channels, self.pad_channels, self.consecutive_channels, self.ranges, max_consecutive_channels=self.max_consecutive_channels)
            # update the mask accordingly
            self.mask+=mask
            self.spectrum_masked[self.mask] = 0

        # remove features with high positive or negative data values
        MAD=self.compute_MAD(self.spectrum[~self.mask])
        # Mask outlier peaks in the spectrum
        mask=self.mask_outlierpeaks(self.spectrum_masked, self.n_channels, self.pad_channels, self.n_mad*MAD, self.ranges)
        # update the mask accordingly
        self.mask+=mask
        self.spectrum_masked[self.mask] = 0

        # determine the channels with noise
        self.noise = self.spectrum[~self.mask]

        if np.size(self.noise)==0:
            self.rms=np.nan
            self.flag='All flagged'
        else:
            #  determine the noise from the remaining channels
            self.rms = np.sqrt(np.sum(self.noise**2) / np.size(self.noise))

    def get_max_consecutive_channels(self, n_channels, p_limit):
        """

        Determine the maximum number of random consecutive positive/negative channels.
        Calculate the number of consecutive positive or negative channels,
        whose probability of occurring due to random chance in a spectrum
        is less than p_limit.

        Parameters
        ----------
        n_channels : int
            Number of spectral channels.
        p_limit : float
            Maximum probability for consecutive positive/negative channels being
            due to chance.

        Returns
        -------
        consec_channels : int
            Number of consecutive positive/negative channels that have a probability
            less than p_limit to be due to chance.

        """
        for consec_channels in range(2, 30):
            a = np.zeros((consec_channels, consec_channels))
            for i in range(consec_channels - 1):
                a[i, 0] = a[i, i + 1] = 0.5
            a[consec_channels - 1, consec_channels - 1] = 1.0
            if np.linalg.matrix_power(
                    a, n_channels - 1)[0, consec_channels - 1] < p_limit:
                return consec_channels

    def mask_broadpeaks(self, n_channels, pad_channels, consecutive_channels, ranges, max_consecutive_channels=14 ):
        """
        Method used to mask out broad peaks within a spectrum

        Parameters
        ----------
        n_channels : int
            Number of spectral channels
        pad_channels : int
            Number of channels by which an interval (low, upp) gets extended on
            both sides, resulting in (low - pad_channels, upp + pad_channels).
        consecutive_channels: int
            number of consecutive channels in a peak
        ranges : array
            List of intervals [(low, upp), ...] determined to contain peaks.
        max_consecutive_channels : int
            Number of consecutive positive/negative channels that have a
            probability less than p_limit to be due to chance.

        Returns
        -------
        mask : array
            A mask that can be applied to the spectrum to mask broad peaks

        """
        mask_consec = consecutive_channels >= max_consecutive_channels
        if mask_consec.any():
            mask = self.mask_channels(n_channels, ranges[mask_consec], pad_channels=pad_channels)
            return mask
        else:
            return np.zeros(n_channels, dtype='bool')

    def determine_peaks(self, spectrum, peak='both'):
        """
        Find peaks in a spectrum.

        Parameters
        ----------
        spectrum : numpy.ndarray
            Array of the data values of the spectrum.
        peak : 'both' (default), 'positive', 'negative'
            Description of parameter `peak`.
        amp_threshold : float
            Required minimum threshold that at least one data point in a peak
            feature has to exceed.

        Returns
        -------
        consecutive_channels or amp_vals : numpy.ndarray
            If the 'amp_threshold' value is supplied an array with the maximum
            data values of the ranges is returned. Otherwise, the number of
            spectral channels of the ranges is returned.
        ranges : list
            List of intervals [(low, upp), ...] determined to contain peaks.

        """

        if (peak == 'both') or (peak == 'positive'):
            clipped_spectrum = spectrum.clip(max=0)
            # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(
                ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

        if (peak == 'both') or (peak == 'negative'):
            clipped_spectrum = spectrum.clip(min=0)
            # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(
                ([0], np.equal(clipped_spectrum, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            if peak == 'both':
                # Runs start and end where absdiff is 1.
                ranges = np.append(
                    ranges, np.where(absdiff == 1)[0].reshape(-1, 2), axis=0)
            else:
                ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

        sort_indices = np.argsort(ranges[:, 0])
        ranges = ranges[sort_indices]

        consecutive_channels = ranges[:, 1] - ranges[:, 0]
        return consecutive_channels, ranges

    def compute_MAD(self, noise_spectrum):
        """
        Computes the median absolute deviation of a spectrum that is assumed to
        be pure noise. If remove_broad=True then broad peaks will have already
        been masked at this point.

        We restrict the calculation of the MAD to spectral channels with
        negative values, since the positive channels can still contain
        multiple narrow high signal peaks that were not identified during
        remove_broad. The assumption is that negative spikes will be
        sufficiently uncommon that they shouldn't affect the computation of MAD.

        Parameters
        ----------
        noise_spectrum : array
            A spectrum which is assumed to contain mostly noise

        Returns
        -------
        MAD : float
            The median absolute deviation of the spectrum computed from spectral
            channels with negative values (reflected noise)

        """
        negative_indices = (noise_spectrum < 0.0)
        spectrum_negative_values = noise_spectrum[negative_indices]
        reflected_noise = np.concatenate((spectrum_negative_values,np.abs(spectrum_negative_values)))

        return median_absolute_deviation(reflected_noise)

    def mask_outlierpeaks(self, spectrum, n_channels, pad_channels, MADthresh, ranges):
        """
        Method used to mask out outlier high-amplitude peaks within a spectrum

        Parameters
        ----------
        spectrum : numpy.ndarray
            Array of the data values of the spectrum
        n_channels : int
            Number of spectral channels
        pad_channels : int
            Number of channels by which an interval (low, upp) gets extended on
            both sides, resulting in (low - pad_channels, upp + pad_channels).
        MADthresh : float
            n_mad*MAD. The threshold used to determine where the peaks lie
        ranges : array
            List of intervals [(low, upp), ...] determined to contain peaks.

        Returns
        -------
        mask : array
            A mask that can be applied to the spectrum to mask high-amplitude
            peaks

        """
        # determine where there are spikes above MADthresh
        inds_high_amps = np.where(np.abs(spectrum) > MADthresh)[0]

        if inds_high_amps.size > 0:
            # Find the ranges of these spikes and mask them
            inds_ranges = np.digitize(inds_high_amps, ranges[:, 0]) - 1
            ranges = ranges[inds_ranges]
            mask = self.mask_channels(n_channels, ranges, pad_channels=pad_channels)
            return mask
        else:
            # if there aren't any, the spectrum is considered to be noise
            return np.zeros(n_channels, dtype='bool')

    def mask_channels(self, n_channels, ranges, pad_channels=None, remove_intervals=None):
        """
        Determine the 1D boolean mask for a given list of spectral ranges.

        Parameters
        ----------
        n_channels : int
            Number of spectral channels.
        ranges : list
            List of intervals [(low, upp), ...].
        pad_channels : int
            Number of channels by which an interval (low, upp) gets extended on both sides, resulting in (low - pad_channels, upp + pad_channels).
        remove_intervals : type
            Nested list containing info about ranges of the spectrum that should be masked out.

        Returns
        -------
        mask : numpy.ndarray
            1D boolean mask that has 'True' values at the position of the channels contained in ranges.

        """
        mask = np.zeros(n_channels)

        for (lower, upper) in ranges:
            if pad_channels is not None:
                lower = max(0, lower - pad_channels)
                upper = min(n_channels, upper + pad_channels)
            mask[lower:upper] = 1

        if remove_intervals is not None:
            for (low, upp) in remove_intervals:
                mask[low:upp] = 0

        return mask.astype('bool')
