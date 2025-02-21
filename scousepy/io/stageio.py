import pickle
import sys

if sys.version_info.major >= 3:
    proto = 4
else:
    proto = 2

from ..scouse import scouse


class LoadScouseObject:
    @staticmethod
    def load_stage_1(scouseobject: scouse, s1path: str):
        """Load the stage 1 scouse object

        Parameters
        ----------
        scouseobject : scouse
            The scouse object to load the stage 1 data into
        s1path : str
            The path to the stage 1 scouse object

        Returns
        -------
        scouse
            The scouse object with the stage 1 data loaded
        """

        with open(s1path, "rb") as fh:
            (
                scouseobject.completed_stages,
                scouseobject.coverage_config_file_path,
                scouseobject.lenspec,
                scouseobject.saa_dict,
                scouseobject.x,
                scouseobject.xtrim,
                scouseobject.trimids,
                scouseobject.rms_approx,
            ) = pickle.load(fh, fix_imports=True)

        return scouseobject


class SaveScouseObject:
    @staticmethod
    def save_stage_1(scouseobject: scouse, s1path: str):
        """Save the stage 1 scouse object

        Parameters
        ----------
        scouseobject : scouse
            The scouse object to save the stage 1 data into
        s1path : str
            The path to save the stage 1 scouse object

        Returns
        -------
        scouse
            The scouse object with the stage 1 data saved
        """

        with open(s1path, "wb") as fh:
            pickle.dump(
                (
                    scouseobject.completed_stages,
                    scouseobject.coverage_config_file_path,
                    scouseobject.lenspec,
                    scouseobject.saa_dict,
                    scouseobject.x,
                    scouseobject.xtrim,
                    scouseobject.trimids,
                    scouseobject.rms_approx,
                ),
                fh,
                fix_imports=True,
                protocol=proto,
            )

        return scouseobject
