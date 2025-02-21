from ..scouse import scouse


class SAAs(object):
    """
    A class for generating SAAs.
    """

    def __init__(self, scouseobject: scouse):
        self.scouseobject = scouseobject

    def establishSAAs(self):
        """
        Establish the SAAs.

        Parameters
        ----------
        scouseobject : scouse
            The scouse object.
        """

        return self

    def _createSAAcube(self):
        """
        Create the SAA cube.

        Parameters
        ----------
        scouseobject : scouse
            The scouse object.
        """
        pass
