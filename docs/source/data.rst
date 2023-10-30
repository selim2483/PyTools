.. role:: hidden
    :class: hidden-section

pytools.data
============
.. automodule:: pytools.data

Customized utilitaries image datasets.

.. contents:: pytools.data
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: pytools

Augmentations
-------------

Customized data augmentations processes.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    data.augmentations.augment

Datasets
--------

Python datasets utilitaries.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    data.Dataset
    data.ImageFolderDataset
    data.dataset_tool.convert_dataset

Utils 
-----

Miscellaneous utils for data processing.

RandomCrop
^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    data.get_mask_crop
    data.mask_crop.convert_corner

Signal processing
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    data.periodic_smooth_decomposition

