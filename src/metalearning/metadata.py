import pandas as pd
from tqdm import tqdm


class MetaSample:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, identifier, train_dataset, test_dataset, results=None):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self._identifier = identifier
        self._train_dataset = train_dataset
        self._test_dataset = test_dataset
        self._results = results

    @property
    def identifier(self):
        return self._identifier

    @property
    def time_series(self):
        return self._train_dataset

    @property
    def test_time_series(self):
        return self._test_dataset

    @property
    def results(self):
        return self._results

    def metafeatures(self, metafeature_functions):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        metafeatures = pd.Series(
            data=[calc(self.time_series) for calc in metafeature_functions],
            index=[calc.__name__ for calc in metafeature_functions],
        )
        return metafeatures

    def get_best_hyperparameters(self, nr_best):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        best_configs_df = self.results.sort_values(by=[("diagnostics", "mae")]).iloc[
            :nr_best
        ]["hyperparameters"]
        best_configs = [best_configs_df.iloc[i].to_dict() for i in range(nr_best)]
        return best_configs

    def get_best_performance(self):
        """Class methods are similar to regular functions.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1: The first parameter.
            param2: The second parameter.

        Returns:
            True if successful, False otherwise.

        """
        best_perf = self.results.sort_values(by=[("diagnostics", "mae")]).iloc[0][
            ("diagnostics", "mae")
        ]
        return best_perf


class MetaDataset:
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, metasamples, metafeature_functions):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
            param3 (:obj:`list` of :obj:`str`): Description of `param3`.

        """
        self.metasamples = metasamples
        self.metafeature_functions = metafeature_functions
        self.metafeature_set = pd.DataFrame(
            data=[
                metasample.metafeatures(metafeature_functions)
                for metasample in tqdm(
                    self.metasamples, desc="Calculate metafeatures of metasamples"
                )
            ],
            index=[metasample.identifier for metasample in self.metasamples],
        )
